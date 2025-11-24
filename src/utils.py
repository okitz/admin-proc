import os
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_community import GoogleSearchAPIWrapper


def get_llm(temperature: float = 0.0, model_name: str = None) -> BaseChatModel:
    """環境変数に基づいてLLMインスタンスを返すファクトリ関数"""
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    else:
        # Default: Gemini (Google)
        target_model = model_name if model_name else "gemini-2.5-flash-lite"
        return ChatGoogleGenerativeAI(model=target_model, temperature=temperature)


def get_search_tool(k: int = 5, include_domains: list[str] = None):
    """環境変数に基づいて検索ツールを返す"""
    use_google = os.getenv("USE_GOOGLE_SEARCH", "false").lower() == "true"

    if use_google:
        wrapper = GoogleSearchAPIWrapper()

        def google_search(query: str):
            # Google Search API doesn't support include_domains easily via this wrapper in the same way
            return wrapper.results(query, num_results=k)
        return google_search
    else:
        # Default: Tavily
        kwargs = {"max_results": k}
        if include_domains:
            kwargs["include_domains"] = include_domains
        return TavilySearchResults(**kwargs)


def load_prompt(filename: str) -> str:
    """promptsディレクトリからMarkdownファイルを読み込む"""
    prompt_path = Path(__file__).parent / "prompts" / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_personas(filename: str = "personas.txt") -> list[str]:
    """promptsディレクトリからペルソナ定義ファイルを読み込み、リストで返す"""
    path = Path(__file__).parent / "prompts" / filename
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # '---' で分割し、空要素を除去してstripする
    return [p.strip() for p in content.split("---") if p.strip()]
