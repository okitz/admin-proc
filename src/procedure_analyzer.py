from concurrent.futures import ThreadPoolExecutor
from typing import cast

import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph

from config import DEFAULT_SEARCH_K, MAX_GENERATION_RETRIES, MAX_SEARCH_LOOPS, MODEL_FAST, MODEL_HIGH_QUALITY
from schemas import AnalysisResult, EvaluationResult, UrlSelection
from states import ResearchState
from utils import generate_prompt_definitions, get_llm, get_search_tool, load_personas, load_prompt
from validator import GraphValidator


class ProcedureAnalyzer:
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
        workflow.add_node("search", self._search_node)
        workflow.add_node("crawl", self._crawl_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.set_entry_point("search")

        # エッジ定義
        workflow.add_edge("search", "crawl")
        workflow.add_edge("crawl", "evaluate")

        # Evaluate -> (Generate or Search)
        workflow.add_conditional_edges("evaluate", self._check_sufficiency, {"generate": "generate", "search": "search"})

        # Generate -> Validate
        workflow.add_edge("generate", "validate")

        # Validate -> (Generate or End)
        workflow.add_conditional_edges("validate", self._check_validation, {"generate": "generate", END: END})

        return workflow.compile()

    # 条件分岐ロジック
    def _check_sufficiency(self, state: ResearchState):
        if state["is_sufficient"]:
            return "generate"
        elif state["search_loop_count"] > self.max_search_loops:  # 最大ループ回数
            print("Max loop count reached. Proceeding to generation anyway.")
            return "generate"
        else:
            return "search"  # 再検索へ

    def _check_validation(self, state: ResearchState):
        if not state.get("validation_error"):
            return END
        elif state.get("generation_retry_count", 0) >= self.max_generation_retries:  # 最大リトライ回数
            print("Max retry count reached. Outputting invalid result.")
            return END
        else:
            print("Validation failed. Retrying generation...")
            return "generate"

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
                response = requests.head(url, timeout=5, allow_redirects=True)
                # 2xxステータスコードなら有効
                if response.status_code < 300:
                    return url
            except requests.RequestException:
                # タイムアウト、接続エラーなど
                pass
            return None

        valid_urls = set()
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(check_url_availability, res["url"]): res for res in pre_filtered_results}
            for future in future_to_url:
                result = future.result()
                if result:
                    valid_urls.add(result)

        return [res for res in pre_filtered_results if res["url"] in valid_urls]

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

        return {"search_queries": [query], "collected_texts": [formatted_results], "candidate_urls": candidate_urls}

    def _normalize_url(self, url: str) -> str:
        """URLを正規化する（末尾のスラッシュを削除）"""
        return url.rstrip("/")

    def _crawl_node(self, state: ResearchState):
        """有望なURLを選定し、スクレイピングする"""
        print("--- Select & Crawl Node ---")
        prompt_tmpl = load_prompt("judge_url.md")
        search_results_text = state["collected_texts"][-1]
        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            search_results=search_results_text,
            missing_info=state.get("missing_info", "特にありません。まずは全体像を把握してください。"),
        )

        structured_llm = cast(Runnable[str, UrlSelection], self.llm_fast.with_structured_output(UrlSelection))

        # URL選定
        try:
            selection = structured_llm.invoke(prompt)
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
            loader = WebBaseLoader(web_path=target_url, requests_per_second=5, requests_kwargs={"timeout": 10})
            docs = loader.load()
            page_content = docs[0].page_content
            print(f"Crawled {len(page_content)} chars.")
        except Exception as e:
            print(f"Failed to crawl target URL: {e}")
            page_content = f"Error: Failed to crawl {target_url}."

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

        full_text = "\n\n".join(state["collected_texts"])
        last_crawled_url = state["visited_urls"][-1]
        last_crawled_text = state["collected_texts"][-1]

        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            last_crawled_url=last_crawled_url,
            last_crawled_text=last_crawled_text,
            previous_missing_info=state.get("missing_info", "特にありません。"),
            full_text=full_text[:50000],
        )

        structured_llm = cast(Runnable[str, EvaluationResult], self.llm_fast.with_structured_output(EvaluationResult))

        try:
            evaluation = structured_llm.invoke(prompt)
            # is_relevantがFalseと判定された場合、クロールした情報が無関係だったことを意味する
            # その場合、収集したテキストと訪問済みURLから最後の要素を削除する
            if not evaluation.is_relevant:
                print("Last crawled content is not relevant. Discarding.")
                state["collected_texts"].pop()
                state["visited_urls"].pop()

            is_relevant = evaluation.is_relevant
            is_sufficient = evaluation.is_sufficient
            missing_info = evaluation.missing_info
            print(f"Is Relevant: {is_relevant}, Is Sufficient: {is_sufficient}, Missing: {missing_info}")

            return {"is_sufficient": is_sufficient, "missing_info": missing_info}
        except Exception as e:
            error_message = f"評価中にエラーが発生しました: {e}"
            print(error_message)
            # エラー時はクロールしたテキストのみを破棄し、訪問済みURLは記録に残す
            # これにより、同じURLへの再クロールを防ぐ
            state["collected_texts"].pop()
            return {"is_sufficient": False, "missing_info": error_message}

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
        result = structured_llm.invoke(formatted_prompt)

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
