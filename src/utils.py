import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.0, model_name: str = "") -> BaseChatModel:
    """環境変数に基づいてLLMインスタンスを返すファクトリ関数"""
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    else:
        # Default: Gemini (Google)
        target_model = model_name if model_name else "gemini-2.5-flash-lite"
        return ChatGoogleGenerativeAI(model=target_model, temperature=temperature)


def get_search_tool(k: int = 5, include_domains: list[str] | None = None) -> Runnable:
    """環境変数に基づいて検索ツールを返す"""
    use_google = os.getenv("USE_GOOGLE_SEARCH", "false").lower() == "true"

    if use_google:
        wrapper = GoogleSearchAPIWrapper()

        def google_search_logic(query: str) -> str:
            """Google Searchを実行し、Tavilyと同様の形式の文字列で結果を返す"""
            search_query = query
            if include_domains:
                site_restriction = " OR ".join(f"site:{domain}" for domain in include_domains)
                search_query = f"{query} ({site_restriction})"

            results = wrapper.results(search_query, num_results=k)
            return "\n\n".join([f"[{res['title']}]({res['link']})\n{res['snippet']}" for res in results])

        return RunnableLambda(google_search_logic)
    else:
        # Default: Tavily
        # 実行時にインポートする
        from langchain_community.tools.tavily_search import TavilySearchResults

        kwargs: dict[str, Any] = {"max_results": k}
        if include_domains:
            kwargs["include_domains"] = include_domains
        return TavilySearchResults(**kwargs)


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
