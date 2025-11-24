from typing import Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from states import ResearchState
from schemas import AnalysisResult, UrlSelection, EvaluationResult
from utils import get_llm, get_search_tool, load_prompt, load_personas
from config import MODEL_FAST, MODEL_HIGH_QUALITY
from validator import GraphValidator

class ProcedureAnalyzer:
    def __init__(self, model_name: str = MODEL_FAST, high_quality_model_name: str = MODEL_HIGH_QUALITY, use_google_search: bool = False):
        """
        コンストラクタで「道具（LLM, ツール）」と「グラフ構造」を確定させる
        """
        # 1. コンポーネントの初期化
        # Search/Crawl/Evaluate用
        self.llm_fast = get_llm(model_name=model_name)
        # Generate用
        self.llm_high_quality = get_llm(model_name=high_quality_model_name)
        
        # 検索ツール (utils.get_search_toolは内部で環境変数を読むが、引数で制御できるように拡張しても良い)
        # ここでは既存のutilsの実装に合わせて呼び出す
        self.search_tool_factory = get_search_tool # 検索クエリごとに呼び出すためファクトリとして保持、あるいは都度呼び出し
        
        # 2. グラフのビルドとコンパイル（1回だけ実行）
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
        
        # エントリーポイント
        workflow.set_entry_point("search")
        
        # エッジ定義
        workflow.add_edge("search", "crawl")
        workflow.add_edge("crawl", "evaluate")
        
        # Evaluate -> (Generate or Search)
        workflow.add_conditional_edges(
            "evaluate",
            self._check_sufficiency,
            {"generate": "generate", "search": "search"}
        )
        
        # Generate -> Validate
        workflow.add_edge("generate", "validate")
        
        # Validate -> (Generate or End)
        workflow.add_conditional_edges(
            "validate",
            self._check_validation,
            {"generate": "generate", END: END}
        )
        
        return workflow.compile()

    # --- Conditional Logic ---
    def _check_sufficiency(self, state: ResearchState):
        if state["is_sufficient"]:
            return "generate"
        elif state["loop_count"] > 3: # 最大ループ回数
            print("Max loop count reached. Proceeding to generation anyway.")
            return "generate"
        else:
            return "search" # 再検索へ戻る

    def _check_validation(self, state: ResearchState):
        if state.get("validation_error"):
            if state.get("retry_count", 0) < 3: # 最大リトライ回数
                print("Validation failed. Retrying generation...")
                return "generate"
            else:
                print("Max retry count reached. Outputting invalid result.")
                return END
        return END

    # --- Node Implementations ---
    def _search_node(self, state: ResearchState):
        """検索を実行し、URL候補を挙げる"""
        print(f"--- Search Node (Loop: {state['loop_count']}) ---")
        
        results_text = []

        include_domains = None
        query = ""
        if state["loop_count"] == 0:
            include_domains = ["app.oss.myna.go.jp"]
            query = f"{state['city_name']} {state['target_procedure']} 電子申請"
        elif state["loop_count"] == 1:
            query = f"{state['city_name']} {state['target_procedure']}"
        else:
            query = f"{state['city_name']} {state['target_procedure']} {state.get('missing_info', '')}"
        
        # 検索ツールの取得 (都度パラメータを変える可能性があるためここで取得)
        search_tool = get_search_tool(k=10, include_domains=include_domains) # configのDEFAULT_SEARCH_Kを使うべきだが一旦ハードコードまたは引数化
        print(f"Searching for: {query}")
        results = search_tool.invoke(query)
        if results:
            results_text.append(f"--- Search Results for '{query}' ---\n{results}")
        
        return {"search_queries": [query], "collected_texts": ["\n\n".join(results_text)]}

    def _crawl_node(self, state: ResearchState):
        """有望なURLを選定し、スクレイピングする"""
        print("--- Select & Crawl Node ---")
        # self.llm_fast を使用
        prompt_tmpl = load_prompt("judge_url.md")
        
        last_search_result = state["collected_texts"][-1]
        
        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            search_results=last_search_result
        )
        
        structured_llm = self.llm_fast.with_structured_output(UrlSelection)
        
        target_url = None
        try:
            selection = structured_llm.invoke(prompt)
            target_url = selection.url
            reason = selection.reason
            print(f"Selected URL: {target_url} (Reason: {reason})")
        except Exception as e:
            print(f"Error in URL selection: {e}")
            return {"loop_count": state["loop_count"] + 1}

        if not target_url or target_url in state["visited_urls"]:
            print("URL already visited or invalid. Skipping.")
            return {"loop_count": state["loop_count"] + 1}

        # Crawling
        from langchain_community.document_loaders import WebBaseLoader
        try:
            loader = WebBaseLoader(target_url)
            docs = loader.load()
            page_content = docs[0].page_content
            print(f"Crawled {len(page_content)} chars from {target_url}")
        except Exception as e:
            print(f"Failed to crawl {target_url}: {e}")
            page_content = "Error: Failed to crawl."

        formatted_text = f"<SOURCE url='{target_url}'>\n{page_content}\n</SOURCE>"
        return {
            "collected_texts": [formatted_text],
            "visited_urls": [target_url],
            "loop_count": state["loop_count"] + 1
        }

    def _evaluate_node(self, state: ResearchState):
        """情報の充足度を判定する"""
        print("--- Evaluate Node ---")
        # self.llm_fast を使用
        prompt_tmpl = load_prompt("evaluate_info.md")
        
        full_text = "\n\n".join(state["collected_texts"])
        
        prompt = prompt_tmpl.format(
            city_name=state["city_name"],
            target_procedure=state["target_procedure"],
            collected_text=full_text[:50000]
        )
        
        structured_llm = self.llm_fast.with_structured_output(EvaluationResult)
        
        try:
            evaluation = structured_llm.invoke(prompt)
            is_sufficient = evaluation.is_sufficient
            missing_info = evaluation.missing_info
            print(f"Is Sufficient: {is_sufficient}, Missing: {missing_info}")
            
            return {
                "is_sufficient": is_sufficient,
                "missing_info": missing_info
            }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {"is_sufficient": False}

    def _generate_node(self, state: ResearchState):
        """最終的なグラフ生成を行う"""
        print("--- Generate Node ---")
        # self.llm_high_quality を使用
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
            input_text=input_text
        )

        structured_llm = self.llm_high_quality.with_structured_output(AnalysisResult)
        result = structured_llm.invoke(formatted_prompt)

        return {"final_output": result.model_dump(), "validation_error": None}

    def _validate_node(self, state: ResearchState):
        """生成されたグラフを検証する"""
        print("--- Validate Node ---")
        result = state["final_output"]
        if not result:
            return {"validation_error": "No output generated."}

        graph_data = {
            "nodes": result["analog_nodes"] + result["digital_nodes"],
            "edges": result["analog_edges"] + result["digital_edges"]
        }
        
        validator = GraphValidator(graph_data)
        is_valid, errors, warnings = validator.validate()
        
        if warnings:
            print(f"Validation Warnings: {warnings}")
            
        if not is_valid:
            error_msg = "\n".join(errors)
            print(f"Validation Failed:\n{error_msg}")
            return {
                "validation_error": error_msg,
                "retry_count": state.get("retry_count", 0) + 1
            }
        
        print("Validation Passed.")
        return {"validation_error": None}

    def run(self, city_name: str, target_procedure: str, config: Optional[dict] = None):
        """外部から呼び出す実行メソッド"""
        initial_state = ResearchState(
            city_name=city_name,
            target_procedure=target_procedure,
            search_queries=[],
            visited_urls=[],
            collected_texts=[],
            loop_count=0,
            is_sufficient=False,
            missing_info=None,
            final_output=None,
            validation_error=None,
            retry_count=0
        )
        return self.app.invoke(initial_state, config=config)
