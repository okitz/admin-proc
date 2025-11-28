import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """LangGraphの状態管理用TypedDict"""

    city_name: str
    target_procedure: str

    # 検索・収集フェーズ用
    search_queries: list[str]
    visited_urls: list[str]
    collected_texts: Annotated[list[str], operator.add]  # 追記型

    # 制御用
    loop_count: int
    is_sufficient: bool
    missing_info: str | None

    # 最終結果
    final_output: dict | None  # AnalysisResultをdict化したもの
    validation_error: str | None  # バリデーションエラーメッセージ
    retry_count: int  # 再生成回数
