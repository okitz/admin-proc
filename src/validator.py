from typing import Any

import networkx as nx


class GraphValidator:
    def __init__(self, graph_data: dict[str, Any]):
        """
        graph_data expects:
        {
            "nodes": [{"id": "...", "type": "...", "label": "..."}, ...],
            "edges": [{"source_id": "...", "target_id": "...", "type": "...", ...}, ...]
        }
        """
        self.nodes = graph_data.get("nodes", [])
        self.edges = graph_data.get("edges", [])
        self.node_map = {n["id"]: n for n in self.nodes}
        self.errors = []
        self.warnings = []

        # NetworkXグラフの構築（トポロジーチェック用）
        self.G = nx.DiGraph()
        for n in self.nodes:
            self.G.add_node(n["id"], type=n["type"])
        for e in self.edges:
            if e["source_id"] in self.node_map and e["target_id"] in self.node_map:
                self.G.add_edge(e["source_id"], e["target_id"], type=e["type"])

    def validate(self) -> tuple[bool, list[str], list[str]]:
        """実行関数。TrueならCritical Errorなし。"""
        self._check_ids()
        if self.errors:
            return False, self.errors, self.warnings  # IDエラーなら即終了

        self._check_connectivity()
        self._check_cycles()
        self._check_type_constraints()
        self._check_domain_logic()

        return len(self.errors) == 0, self.errors, self.warnings

    def _check_ids(self):
        """IDの存在確認"""
        node_ids = set(self.node_map.keys())
        for i, edge in enumerate(self.edges):
            if edge["source_id"] not in node_ids:
                self.errors.append(f"Edge[{i}]: Source ID '{edge['source_id']}' not found in nodes.")
            if edge["target_id"] not in node_ids:
                self.errors.append(f"Edge[{i}]: Target ID '{edge['target_id']}' not found in nodes.")

    def _check_connectivity(self):
        """
        到達可能性とシンクノードのチェック
        ルール:
        1. Stateタイプは終端になりうる。
        2. Physical_Artifact（認定通知書など）は終端になりうる。
        3. Digital_Object は原則として終端になってはいけない（Submitされるべき）。
        ただし、その生成元がすべて 'No_Action' である場合に限り、
        「システム連携済み」とみなして終端であることを許容する。
        """
        # NetworkXのグラフGは __init__ で作成済みと仮定
        # 出次数が0のノード（Sink）を抽出
        sinks = [n for n, d in self.G.out_degree() if d == 0]

        for sink_id in sinks:
            node = self.node_map[sink_id]
            node_type = node["type"]

            # 入ってくるエッジ（In-edges）のタイプを取得
            in_edges = list(self.G.in_edges(sink_id))
            incoming_types = []
            for u, v in in_edges:
                edge_data = self.G.get_edge_data(u, v)
                incoming_types.append(edge_data["type"])

            # --- ケース1: デジタルデータ (前回議論した部分) ---
            if node_type == "Digital_Object":
                # すべてが No_Action 由来ならOK、それ以外はNG
                if incoming_types and all(t == "No_Action" for t in incoming_types):
                    continue
                self.warnings.append(f"Dangling Data Warning: Digital Object '{node['label']}' is created but never submitted.")

            # --- ケース2: 物理的成果物 (今回追加する部分) ---
            elif node_type == "Physical_Artifact":
                # 許容される終端: 行政からの結果通知 (Wait_Result等の結果として得られたもの)
                # ※Wait_ResultのTargetはState推奨ですが、Artifactの場合も想定
                if "Wait_Result" in incoming_types:
                    continue

                # 許容されない終端: ユーザーが能動的に作ったもの
                # (Fill, Copy, Acquire, Print, Enclose で作られたものは、提出されないとおかしい)
                user_action_types = [
                    "Physical_Fill",
                    "Physical_Copy_Store",
                    "Physical_Print_Store",
                    "Physical_Acquire_Local",
                    "External_Acquire",
                    "Physical_Enclose",
                ]

                if any(t in user_action_types for t in incoming_types):
                    self.errors.append(  # これはWarningではなくErrorで良いレベル
                        f"Unsubmitted Artifact Error: '{node['label']}' was prepared but not Enclosed or Submitted."
                    )

            # --- ケース3: 状態 (State) ---
            elif node_type == "State":
                continue  # 正常な終了

            # --- ケース4: その他 (Raw_Materialなど) ---
            else:
                self.warnings.append(f"Dead-end Node Warning: '{node['label']}' [{node_type}] should not be a sink.")

    def _check_cycles(self):
        """循環参照のチェック"""
        try:
            cycle = nx.find_cycle(self.G, orientation="original")
            self.errors.append(f"Cycle detected in graph: {cycle}. Procedure must be DAG.")
        except nx.NetworkXNoCycle:
            pass

    def _check_type_constraints(self):
        """エッジの型制約チェック（Schemaで定義したルール）"""
        # 許容される SourceType -> EdgeType -> TargetType の定義
        RULES = {
            # Physical Acquire & Transform
            "Physical_Possess": ("State", ["Physical_Artifact", "Raw_Material"]),
            "Physical_Acquire_Resident": ("State", "Physical_Artifact"),
            "External_Acquire": ("State", "Physical_Artifact"),
            "Physical_Print_Store": ("Digital_Object", "Raw_Material"),
            "Physical_Print_Home": ("Digital_Object", "Raw_Material"),
            "Physical_Get_Material": ("State", "Raw_Material"),
            "Physical_Fill": ("Raw_Material", "Physical_Artifact"),
            "Physical_Copy_Store": ("Physical_Artifact", "Physical_Artifact"),
            "Physical_Enclose": (["Physical_Artifact", "Raw_Material"], "Physical_Artifact"),
            # Physical Submit
            "Physical_Submit_Window": ("Physical_Artifact", "State"),
            "Physical_Submit_Mail": ("Physical_Artifact", "State"),
            # Digital
            "Digital_Access": ("State", "System"),
            "Digital_Download": ("System", "Digital_Object"),
            "Digital_Auth": ("System", "System"),
            "Digital_Input": ("System", "Digital_Object"),
            "Digital_Capture": ("Physical_Artifact", "Digital_Object"),
            "Digital_Upload": ("Digital_Object", "Digital_Object"),
            "Digital_Submit": ("Digital_Object", "State"),
            # Skip / Wait
            "No_Action": ("System", "Digital_Object"),
            "Wait_Result": ("State", "State"),
        }

        for e in self.edges:
            src_type = self.node_map[e["source_id"]]["type"]
            tgt_type = self.node_map[e["target_id"]]["type"]
            edge_type = e["type"]

            if edge_type in RULES:
                valid_src, valid_tgt = RULES[edge_type]
                # Sourceチェック
                if isinstance(valid_src, list):
                    if src_type not in valid_src:
                        self.errors.append(f"Type Error: Edge '{edge_type}' cannot start from '{src_type}' ({e['source_id']}).")
                elif src_type != valid_src:
                    self.errors.append(
                        f"Type Error: Edge '{edge_type}' cannot start from '{src_type}' "
                        f"({e['source_id']}). Expected {valid_src}."
                    )

                # Targetチェック
                if tgt_type != valid_tgt:
                    self.errors.append(
                        f"Type Error: Edge '{edge_type}' cannot end at '{tgt_type}' ({e['target_id']}). Expected {valid_tgt}."
                    )

    def _check_domain_logic(self):
        """ドメイン特有の論理チェック"""

        # 1. アナログ申請の「無からの発生」チェック
        # Physical_Fill (記入) があるのに、その親に Physical_Print や Physical_Get_Material がない
        fill_edges = [e for e in self.edges if e["type"] == "Physical_Fill"]
        for fill in fill_edges:
            paper_id = fill["source_id"]
            # その紙(Raw_Material)を作ったエッジを探す
            predecessors = list(self.G.predecessors(paper_id))
            if not predecessors:
                self.warnings.append(f"Logic Warning: Paper '{paper_id}' is filled but has no origin (Print/Get).")

        # 2. デジタル申請の「多重送信」チェック
        # Digital_Submit が複数ある場合、統合されていない可能性がある
        submit_edges = [e for e in self.edges if e["type"] == "Digital_Submit"]
        if len(submit_edges) > 1:
            self.warnings.append(
                f"Logic Warning: Multiple Digital_Submit edges found ({len(submit_edges)}). Data might not be aggregated."
            )

        # 3. Evidenceの欠落チェック
        for e in self.edges:
            if not e.get("evidence_text"):
                self.warnings.append(f"Evidence Warning: Edge '{e['action']}' has no evidence text.")


if __name__ == "__main__":
    # ダミーデータ
    data = {
        "nodes": [{"id": "d4", "type": "System", "label": "画面"}, {"id": "d9", "type": "State", "label": "完了"}],
        "edges": [
            {"source_id": "d4", "target_id": "d9", "type": "Digital_Submit", "action": "送信", "evidence_text": "implicit"}
        ],
    }

    validator = GraphValidator(data)
    is_valid, errs, warns = validator.validate()

    print(f"Valid: {is_valid}")
    print("Errors:", errs)  # Digital_SubmitはObject->StateなのでType Errorが出るはず
    print("Warnings:", warns)
