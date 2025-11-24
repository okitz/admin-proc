from typing import TypedDict, Annotated, List, Optional
import operator


class ResearchState(TypedDict):
    """LangGraphの状態管理用TypedDict"""
    city_name: str
    target_procedure: str

    # 検索・収集フェーズ用
    search_queries: List[str]
    visited_urls: List[str]
    collected_texts: Annotated[List[str], operator.add]  # 追記型

    # 制御用
    loop_count: int
    is_sufficient: bool
    missing_info: Optional[str]

    # 最終結果
    final_output: Optional[dict]  # AnalysisResultをdict化したもの
    validation_error: Optional[str]  # バリデーションエラーメッセージ
    retry_count: int  # 再生成回数
