import datetime
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar, cast

import requests
from bs4 import BeautifulSoup
from google.api_core.retry import Retry
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph
from requests.exceptions import RequestException

from agent.state import ResearchState
from agent.utils import generate_prompt_definitions, get_llm, get_search_tool, load_personas, load_prompt
from agent.validator import GraphValidator
from core.config import DEFAULT_SEARCH_K, MAX_GENERATION_RETRIES, MAX_SEARCH_LOOPS, MODEL_FAST, MODEL_HIGH_QUALITY
from core.schemas import AnalysisResult, EvaluationResult, UrlSelection

_T = TypeVar("_T")


class ProcedureAnalyzer:
    # --- Constants ---
    # Node Names
    _NODE_SEARCH = "search"
    _NODE_CRAWL = "crawl"
    _NODE_EVALUATE = "evaluate"
    _NODE_GENERATE = "generate"
    _NODE_VALIDATE = "validate"

    # Conditional Edges
    _EDGE_GENERATE = "generate"
    _EDGE_SEARCH = "search"

    # Misc
    _URL_CHECK_TIMEOUT = 5
    _CRAWL_TIMEOUT = 10
    _MAX_CONTENT_SIZE = 1_000_000  # 1MB

    def __init__(
        self,
        fast_model_name: str = MODEL_FAST,
        high_quality_model_name: str = MODEL_HIGH_QUALITY,
        max_search_loops: int = MAX_SEARCH_LOOPS,
        max_generation_retries: int = MAX_GENERATION_RETRIES,
    ):
        """
        コンストラクタでモデル、グラフの定義
        """
        self.llm_fast = get_llm(model_name=fast_model_name)
        self.llm_high_quality = get_llm(model_name=high_quality_model_name)
        self.max_search_loops = max_search_loops
        self.max_generation_retries = max_generation_retries

        self.app = self._build_workflow()

    def _build_workflow(self):
        """内部メソッド: ノードとエッジを定義してコンパイルする"""
        workflow = StateGraph(ResearchState)

        # ノード定義
        workflow.add_node(self._NODE_SEARCH, self._search_node)
        workflow.add_node(self._NODE_CRAWL, self._crawl_node)
        workflow.add_node(self._NODE_EVALUATE, self._evaluate_node)
        workflow.add_node(self._NODE_GENERATE, self._generate_node)
        workflow.add_node(self._NODE_VALIDATE, self._validate_node)
        workflow.set_entry_point(self._NODE_SEARCH)

        # エッジ定義
        workflow.add_edge(self._NODE_SEARCH, self._NODE_CRAWL)
        workflow.add_edge(self._NODE_CRAWL, self._NODE_EVALUATE)

        # Evaluate -> (Generate or Search)
        workflow.add_conditional_edges(
            self._NODE_EVALUATE,
            self._check_sufficiency,
            {self._EDGE_GENERATE: self._NODE_GENERATE, self._EDGE_SEARCH: self._NODE_SEARCH},
        )

        # Generate -> Validate
        workflow.add_edge(self._NODE_GENERATE, self._NODE_VALIDATE)

        # Validate -> (Generate or End)
        workflow.add_conditional_edges(
            self._NODE_VALIDATE, self._check_validation, {self._EDGE_GENERATE: self._NODE_GENERATE, END: END}
        )

        return workflow.compile()

    # 条件分岐ロジック
    def _check_sufficiency(self, state: ResearchState):
        if state["is_sufficient"]:
            return self._EDGE_GENERATE
        if state["search_loop_count"] >= self.max_search_loops:  # 最大ループ回数
            print("Max search loop count reached. Proceeding to generation anyway.")
            return self._EDGE_GENERATE
        return self._EDGE_SEARCH

    def _check_validation(self, state: ResearchState):
        if not state.get("validation_error"):
            return END
        if state.get("generation_retry_count", 0) >= self.max_generation_retries:  # 最大リトライ回数
            print("Max generation retry count reached. Outputting invalid result.")
            return END
        print("Validation failed. Retrying generation...")
        return self._EDGE_GENERATE

    # --- Helper Methods for Nodes ---

    def _generate_search_query(self, state: ResearchState) -> str:
        """検索ループ回数に応じて検索クエリを生成する"""
        if state["search_loop_count"] == 0:
            return f"{state['city_name']} {state['target_procedure']}"
        return f"{state['city_name']} {state['target_procedure']} {state.get('missing_info', '')}"

    def _filter_and_validate_urls(self, search_results: list[dict], visited_urls: set[str]) -> list[dict]:
        """検索結果をフィルタリングし、URLの有効性を検証する"""
        if not search_results:
            print("No search results found.")
            return []

        # 検索結果から、URLやコンテンツがないもの、訪問済みのものを除外
        pre_filtered_results = [
            res
            for res in search_results
            if res.get("url") and res.get("content") and self._normalize_url(res["url"]) not in visited_urls
        ]

        # URLの有効性を実際に確認する (404エラーなどを除外)
        def check_url_availability(url: str) -> str | None:
            try:
                # HEADリクエストでヘッダーのみ取得。タイムアウトを設定。
                response = requests.head(url, timeout=self._URL_CHECK_TIMEOUT, allow_redirects=True)
                # 2xxステータスコードなら有効
                if response.status_code < 300:
                    return url
            except RequestException:
                # タイムアウト、接続エラーなど
                pass
            return None

        valid_urls = set()
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(check_url_availability, self._normalize_url(res["url"])): res for res in pre_filtered_results
            }
            for future in future_to_url:
                result = future.result()
                if result:
                    valid_urls.add(result)

        return [res for res in pre_filtered_results if self._normalize_url(res["url"]) in valid_urls]

    def _format_search_results_for_llm(self, query: str, results: list[dict]) -> str:
        """検索結果をLLMのプロンプト用にフォーマットする"""
        if not results:
            return f"--- Search Results for '{query}' ---\n(No results found)"

        results_str_list = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            url = r.get("url", "")
            content = r.get("content", "")[:300].replace("\n", " ")
            results_str_list.append(f"[{i}] {title}\nURL: {url}\nSummary: {content}")
        return f"--- Search Results for '{query}' ---\n" + "\n".join(results_str_list)

    def _save_prompt_to_file(self, node_name: str, prompt: str) -> Path:
        """Saves the given prompt to a file for debugging/auditing."""
        log_dir = Path(__file__).parent.parent.parent / "prompt_logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = log_dir / f"{timestamp}_{node_name}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        return filepath

    def _clean_html_content(self, html_content: str) -> str:
        """HTMLコンテンツから不要なタグと空白を削除してテキストを抽出する"""
        soup = BeautifulSoup(html_content, "html.parser")

        # 不要なタグ（スクリプト、スタイル、ナビゲーション、ヘッダー、フッター等）を削除
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # テキストを取得
        text = soup.get_text(separator="\n")

        # 複数のスペースやタブを単一のスペースに置換
        text = re.sub(r"[ \t]+", " ", text)
        # 3行以上の連続した改行を2行の改行に置換
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 各行の先頭と末尾の空白を削除し、空行が複数続かないようにする
        lines = [line.strip() for line in text.splitlines()]
        cleaned_text = "\n".join(line for line in lines if line)

        return cleaned_text

    @Retry(initial=10, maximum=100, multiplier=2, timeout=200)
    def _call_llm(self, structured_llm: Runnable[str, _T], prompt: str, node_name: str) -> _T:
        """LLMを呼び出す共通メソッド。プロンプト保存とリトライ処理を内包する。"""
        self._save_prompt_to_file(node_name, prompt)
        return structured_llm.invoke(prompt)

    # --- Node Implementations ---

    def _search_node(self, state: ResearchState):
        """検索を実行し、URL候補を挙げる"""
        print(f"--- Search Node (Loop: {state['search_loop_count']}) ---")
        query = self._generate_search_query(state)
        search_tool = get_search_tool(k=DEFAULT_SEARCH_K)
        search_results = search_tool.invoke(query)

        visited_urls = {self._normalize_url(u) for u in state.get("visited_urls", [])}
        filtered_results = self._filter_and_validate_urls(search_results, visited_urls)
        excluded_count = len(search_results) - len(filtered_results)

        print(
            f"Search found {len(search_results)} results. "
            f"Filtered out {excluded_count} invalid or visited URLs, leaving {len(filtered_results)} candidates."
        )

        candidate_urls = [res["url"] for res in filtered_results]
        formatted_results = self._format_search_results_for_llm(query, filtered_results)

        return {
            "search_queries": [query],
            "search_results_summary": formatted_results,
            "candidate_urls": candidate_urls,
        }

    def _normalize_url(self, url: str) -> str:
        """URLを正規化する（末尾のスラッシュを削除）"""
        return url.rstrip("/")

    def _crawl_node(self, state: ResearchState):
        """有望なURLを選定し、スクレイピングする"""
        print("--- Select & Crawl Node ---")
        prompt_tmpl = load_prompt("judge_url.md")
        search_results_text = state["search_results_summary"]
        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            search_results=search_results_text,
            missing_info=state.get("missing_info") or "特にありません。まずは全体像を把握してください。",
        )

        structured_llm = cast(Runnable[str, UrlSelection], self.llm_fast.with_structured_output(UrlSelection))

        # URL選定
        try:
            selection = self._call_llm(structured_llm, prompt, "crawl_select_url")
            selected_id = selection.id
            reason = selection.reason
            print(f"LLM selected ID: {selected_id} (Reason: {reason})")
        except Exception as e:
            print(f"Error in URL selection: {e}")
            return {"search_loop_count": state["search_loop_count"] + 1}

        candidate_urls = state.get("candidate_urls", [])
        # IDからURLを取得 (IDは1始まりなので、インデックスは-1する)
        if not (1 <= selected_id <= len(candidate_urls)):
            print(f"LLM selected an invalid ID: {selected_id}. Skipping.")
            return {"search_loop_count": state["search_loop_count"] + 1}

        target_url = self._normalize_url(candidate_urls[selected_id - 1])
        print(f"Target URL: {target_url}")

        # Crawling
        try:
            response = requests.get(target_url, timeout=self._CRAWL_TIMEOUT)
            response.raise_for_status()
            # 文字化け防止のため、エンコーディングを推定して設定
            # コンテンツサイズが大きすぎる場合は処理を中断
            if len(response.content) > self._MAX_CONTENT_SIZE:
                print(f"Content size exceeds limit of {self._MAX_CONTENT_SIZE} bytes. Skipping.")
                page_content = f"Error: Content too large to process from {target_url}."
                return {"collected_texts": [f"<SOURCE url='{target_url}'>\n{page_content}\n</SOURCE>"]}

            response.encoding = response.apparent_encoding
            page_content = self._clean_html_content(response.text)
            print(f"Crawled {len(page_content)} chars.")
        except Exception as e:
            print(f"Failed to crawl target URL: {e}")
            page_content = f"Error: Failed to crawl or clean content from {target_url}."

        formatted_text = f"<SOURCE url='{target_url}'>\n{page_content}\n</SOURCE>"

        return {
            "collected_texts": [formatted_text],
            "visited_urls": [target_url],
            "search_loop_count": state["search_loop_count"] + 1,
        }

    def _evaluate_node(self, state: ResearchState):
        """情報の充足度を判定する"""
        print("--- Evaluate Node ---")
        prompt_tmpl = load_prompt("evaluate_info.md")

        if not state["collected_texts"]:
            # 通常は発生しないが、念のためガード
            return {"is_sufficient": False, "missing_info": "収集されたテキストがありません。"}

        last_crawled_text = state["collected_texts"][-1]
        last_crawled_url = state["visited_urls"][-1]
        previous_texts = "\n\n".join(state["collected_texts"][:-1])

        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            last_crawled_url=last_crawled_url,
            last_crawled_text=last_crawled_text,
            previous_texts=previous_texts[:50000],  # トークン数削減のため制限
            previous_missing_info=state.get("missing_info") or "特にありません。",
        )

        structured_llm = cast(Runnable[str, EvaluationResult], self.llm_fast.with_structured_output(EvaluationResult))

        try:
            evaluation = self._call_llm(structured_llm, prompt, "evaluate_info")
            # is_relevantがFalseと判定された場合、クロールした情報が無関係だったことを意味する
            # その場合、収集したテキストと訪問済みURLから最後の要素を削除する
            if not evaluation.is_relevant:
                print("Last crawled content is not relevant. Discarding and rolling back state.")
                return {
                    "collected_texts": state["collected_texts"][:-1],
                    "visited_urls": state["visited_urls"][:-1],
                    "is_sufficient": False,
                    "missing_info": state.get("missing_info"),  # 以前のmissing_infoを維持
                }
            # LLMの出力がNoneの場合に備えて、ここでデフォルト値を設定
            missing_info_to_store = evaluation.missing_info or "特にありません。"

            print(
                f"Is Relevant: {evaluation.is_relevant}, Is Sufficient: {evaluation.is_sufficient}, "
                f"Missing: {missing_info_to_store}"
            )

            return {"is_sufficient": evaluation.is_sufficient, "missing_info": missing_info_to_store}
        except Exception as e:
            print(f"評価中にエラーが発生しました: {e}")  # デバッグ用にエラー内容は出力
            # 評価に失敗した場合、このループで行った変更をすべて元に戻し、同じ条件で再検索・再評価できるようにする
            print("Evaluation failed. Rolling back state to retry this loop.")
            return {
                "collected_texts": state["collected_texts"][:-1],
                "visited_urls": state["visited_urls"][:-1],
                "search_loop_count": state["search_loop_count"] - 1,
                "is_sufficient": False,
            }

    def _generate_node(self, state: ResearchState):
        """最終的なグラフ生成を行う"""
        print("--- Generate Node ---")
        prompt_tmpl = load_prompt("extraction.md")

        full_text = "\n\n".join(state["collected_texts"])

        personas = load_personas()
        target_persona = personas[0] if personas else "標準ペルソナ設定なし"

        input_text = full_text[:50000]

        if state.get("validation_error"):
            error_msg = state["validation_error"]
            input_text += f"\n\n# 前回の生成におけるエラー (修正してください):\n{error_msg}"

        formatted_prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            persona=target_persona,
            input_text=input_text,
            type_definitions=generate_prompt_definitions(),
        )

        structured_llm = cast(Runnable[str, AnalysisResult], self.llm_high_quality.with_structured_output(AnalysisResult))
        try:
            result = self._call_llm(structured_llm, formatted_prompt, "generate_graph")
        except Exception as e:
            print(f"Error during graph generation: {e}")
            return {"analysis_result": None, "validation_error": str(e)}

        return {"analysis_result": result.model_dump(), "validation_error": None}

    def _validate_node(self, state: ResearchState):
        """生成されたグラフを検証する"""
        print("--- Validate Node ---")
        result = state["analysis_result"]
        if not result:
            return {"validation_error": "No output generated."}

        graph_data = {
            "nodes": result["analog_nodes"] + result["digital_nodes"],
            "edges": result["analog_edges"] + result["digital_edges"],
        }

        validator = GraphValidator(graph_data)
        is_valid, errors, warnings = validator.validate()

        if warnings:
            print(f"Validation Warnings: {warnings}")

        if not is_valid:
            error_msg = "\n".join(errors)
            print(f"Validation Failed:\n{error_msg}")
            return {"validation_error": error_msg, "generation_retry_count": state.get("generation_retry_count", 0) + 1}

        print("Validation Passed.")
        return {"validation_error": None}

    def run(self, city_name: str, target_procedure: str, config: RunnableConfig | None = None):
        """外部から呼び出す実行メソッド"""
        initial_state = ResearchState(
            city_name=city_name,
            target_procedure=target_procedure,
            search_queries=[],
            search_results_summary=None,
            candidate_urls=[],
            visited_urls=[],
            collected_texts=[],
            search_loop_count=0,
            is_sufficient=False,
            missing_info=None,
            analysis_result=None,
            validation_error=None,
            generation_retry_count=0,
        )
        return self.app.invoke(initial_state, config=config)
