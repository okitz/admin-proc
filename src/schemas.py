from pydantic import BaseModel, Field

from definitions import EdgeType, NodeType

# --- 依存関係グラフ ---


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
