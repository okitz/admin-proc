import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from core.definitions import EDGE_DEFINITIONS, NODE_DESCRIPTIONS


def get_llm(temperature: float = 0.0, model_name: str = "") -> BaseChatModel:
    """環境変数に基づいてLLMインスタンスを返すファクトリ関数"""
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    else:
        # Default: Gemini (Google)
        target_model = model_name if model_name else "gemini-2.5-flash-lite"
        return ChatGoogleGenerativeAI(model=target_model, temperature=temperature, max_retries=0)


def get_search_tool(k: int = 5, include_domains: list[str] | None = None) -> RunnableLambda | TavilySearch:
    """環境変数に基づいて検索ツールを返す"""
    use_google = os.getenv("USE_GOOGLE_SEARCH", "false").lower() == "true"
    if use_google:
        wrapper = GoogleSearchAPIWrapper()

        def _google_search_logic(query: str) -> list[dict[str, str]]:
            """Google Searchを実行し、Tavilyと同様の形式の文字列で結果を返す"""
            search_query = query
            if include_domains:
                site_restriction = " OR ".join(f"site:{domain}" for domain in include_domains)
                search_query = f"{query} ({site_restriction})"
            try:
                raw_results = wrapper.results(search_query, num_results=k)
            except Exception as e:
                print(f"Google Search Error: {e}")
                return []

            normalized_results = []
            for res in raw_results:
                normalized_results.append(
                    {"url": res.get("link", ""), "content": res.get("snippet", ""), "title": res.get("title", "")}
                )
            return normalized_results

        return RunnableLambda(_google_search_logic)
    else:
        # Default: Tavily
        kwargs: dict[str, Any] = {"max_results": k}
        if include_domains:
            kwargs["include_domains"] = include_domains
        tavily_tool = TavilySearch(**kwargs)

        def _tavily_search_logic(query: str) -> list[dict[str, str]]:
            """Tavilyを実行し、Googleとキー名を統一して返す"""
            try:
                # 新しいTavilySearchは入力として {"query": "..."} を期待する
                # 戻り値は {"query":..., "results": [...], "images":...} のような辞書
                response = tavily_tool.invoke({"query": query})

                # エラー文字列が返ってきた場合のガード
                if isinstance(response, str):
                    print(f"Tavily Warning: {response}")
                    return []

                # "results" キーからリストを取得
                raw_results = response.get("results", [])

                normalized_results = []
                for res in raw_results:
                    normalized_results.append(
                        {
                            "url": res.get("url", ""),
                            "content": res.get("content", ""),  # 新クラスでも keyは content
                            "title": res.get("title", ""),
                        }
                    )
                return normalized_results
            except Exception as e:
                print(f"Tavily Search Error: {e}")
                return []

        return RunnableLambda(_tavily_search_logic)


def load_prompt(filename: str) -> str:
    """promptsディレクトリからMarkdownファイルを読み込む"""
    prompt_path = Path(__file__).parent / "prompts" / filename
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def load_personas(filename: str = "personas.txt") -> list[str]:
    """promptsディレクトリからペルソナ定義ファイルを読み込み、リストで返す"""
    path = Path(__file__).parent / "prompts" / filename
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # '---' で分割し、空要素を除去してstripする
    return [p.strip() for p in content.split("---") if p.strip()]


def generate_prompt_definitions() -> str:
    # ノード定義
    node_text = "## 定義：Node Types (ノードの型分類)\n抽出するノードは、必ず以下のいずれかの型に分類すること。\n"
    for node_enum, desc in NODE_DESCRIPTIONS.items():
        node_text += f"- **{node_enum.value}:** {desc}\n"

    # エッジ定義
    edge_text = (
        "\n## 定義：Edge Types (アクションの型と接続制約)\n"
        "各エッジは、アクションの内容に最も合致するTypeを選択し、"
        "**括弧内に示す Source(S) -> Target(T) の接続ルールを厳守**すること。\n"
    )

    for edge_enum, meta in EDGE_DEFINITIONS.items():
        srcs = meta["source"] if isinstance(meta["source"], list) else [meta["source"]]
        tgts = meta["target"] if isinstance(meta["target"], list) else [meta["target"]]

        src_str = " / ".join([s.value for s in srcs])
        tgt_str = " / ".join([t.value for t in tgts])

        edge_text += f"- **{edge_enum.value}** (S:{src_str} -> T:{tgt_str})\n"
        edge_text += f"    - {meta['description']}\n"

    return node_text + edge_text
