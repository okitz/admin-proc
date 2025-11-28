from typing import Literal

from pydantic import BaseModel, Field

# --- 依存関係グラフ ---
NodeType = Literal["State", "System", "Raw_Material", "Physical_Artifact", "Digital_Object"]
EdgeType = Literal[
    "Physical_Acquire_Resident",
    "External_Acquire",
    "Physical_Print_Store",
    "Physical_Print_Home",
    "Physical_Get_Material",
    "Physical_Fill",
    "Physical_Copy_Store",
    "Physical_Enclose",
    "Physical_Submit_Window",
    "Physical_Submit_Mail",
    "Digital_Access",
    "Digital_Download",
    "Digital_Input",
    "Digital_Auth",
    "Digital_Capture",
    "Digital_Upload",
    "Digital_Submit",
    "No_Action",
    "Wait_Result",
]


class Node(BaseModel):
    id: str = Field(..., description="ユニークなID (例: n1, n2)")
    label: str = Field(..., description="ノードの名称 (例: 申請書PDF)")
    type: NodeType = Field(..., description="ノードの分類")


class Edge(BaseModel):
    source_id: str = Field(..., description="始点ノードID")
    target_id: str = Field(..., description="終点ノードID")
    description: str = Field(..., description="アクションの具体的な説明")
    type: EdgeType = Field(..., description="コスト計算用のアクション分類")
    evidence_url: str | None = Field(None, description="根拠となった情報のURL")
    evidence_text: str = Field(..., description="根拠となった記述の抜粋")


# --- 中間出力スキーマ ---
class UrlSelection(BaseModel):
    url: str = Field(..., description="選定されたURL")
    reason: str = Field(..., description="選定理由")


class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(..., description="情報が十分かどうか")
    reason: str = Field(..., description="判定理由")
    missing_info: str | None = Field(None, description="不足している情報")


# --- 最終出力スキーマ ---
class AnalysisResult(BaseModel):
    reasoning: str = Field(..., description="グラフ構築に至るまでの思考プロセスと分析")

    # アナログ申請の構成要素
    analog_nodes: list[Node] = Field(..., description="アナログ申請(郵送・窓口)に登場する全ノード")
    analog_edges: list[Edge] = Field(..., description="アナログ申請の依存関係エッジリスト")

    # デジタル申請の構成要素
    digital_nodes: list[Node] = Field(..., description="デジタル申請(マイナポータル等)に登場する全ノード")
    digital_edges: list[Edge] = Field(..., description="デジタル申請の依存関係エッジリスト")
