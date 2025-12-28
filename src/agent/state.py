import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """LangGraphの状態管理用TypedDict"""

    city_name: str
    target_procedure: str

    # 検索・収集フェーズ用
    search_queries: list[str]
    search_results_summary: str | None  # 検索結果のサマリーテキスト
    candidate_urls: list[str]
    visited_urls: Annotated[list[str], operator.add]
    collected_texts: Annotated[list[str], operator.add]

    # 制御用
    search_loop_count: int
    is_sufficient: bool
    missing_info: str | None

    # 最終結果
    analysis_result: dict | None
    validation_error: str | None
    generation_retry_count: int
