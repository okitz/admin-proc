from typing import Any

import networkx as nx

from core.definitions import EDGE_DEFINITIONS, EdgeType, NodeType


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

        # definitions.py からルールをロード
        self.rules = {}
        for edge_enum, meta in EDGE_DEFINITIONS.items():
            # EdgeType(Enum) をキーとして保存
            self.rules[edge_enum] = (meta["source"], meta["target"])

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
            return False, self.errors, self.warnings  # IDエラーなら構造解析不能なので即終了

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
                self.errors.append(
                    f"Invalid ID Reference at Edge[{i}]: Source ID '{edge['source_id']}' is not defined in the nodes list. "
                    "Please verify that the ID exists in the 'nodes' list."
                )
            if edge["target_id"] not in node_ids:
                self.errors.append(
                    f"Invalid ID Reference at Edge[{i}]: Target ID '{edge['target_id']}' is not defined in the nodes list. "
                    "Please verify that the ID exists in the 'nodes' list."
                )

    def _check_connectivity(self):
        """
        到達可能性とシンクノードのチェック
        ルール:
        1. Process_State は終端になりうる。
        2. Physical_Processed_Artifact（認定通知書など）は終端になりうる。
        3. Digital_Data_Object は原則として終端になってはいけない（Submitされるべき）。
           ただし、その生成元がすべて 'System_Auto_Link_Data' (No_Action) である場合に限り、
           「システム連携済み」とみなして終端であることを許容する。
        """

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

            # --- ケース1: デジタルデータ ---
            if node_type == NodeType.Digital_Data_Object:
                # すべてが No_Action 由来ならOK、それ以外はNG
                if incoming_types and all(t == EdgeType.System_Auto_Link_Data for t in incoming_types):
                    continue
                self.warnings.append(
                    f"Connectivity Warning: The digital object '{node['label']}' ({sink_id}) is created but never used or submitted. "
                    "If this data is required for application, please connect it to 'Digital_Submit_Data' or merge it into the main application data."
                )

            # --- ケース2: 物理的成果物 ---
            elif node_type == NodeType.Physical_Processed_Artifact:
                # 許容される終端: 行政からの結果通知 (Wait_Result等の結果として得られたもの)
                # ※Wait_ResultのTargetはState推奨ですが、Artifactの場合も想定して許容
                if EdgeType.Time_Wait_For_Processing in incoming_types:
                    continue

                # 許容されない終端: ユーザーが能動的に作ったもの
                # (Fill, Copy, Acquire, Print, Enclose で作られたものは、提出されないとおかしい)
                user_action_types = [
                    EdgeType.Physical_Write_or_Process,
                    EdgeType.Physical_Duplicate_Copy,
                    EdgeType.Physical_Print_From_Digital,
                    EdgeType.Physical_Obtain_Required_Item,
                    EdgeType.Physical_Obtain_Raw_Material,  # Acquire系はこれらに集約済み
                    EdgeType.Physical_Combine_or_Package,
                ]

                if any(t in user_action_types for t in incoming_types):
                    self.errors.append(  # これはWarningではなくErrorで良いレベル
                        f"Logical Error: The physical item '{node['label']}' ({sink_id}) was prepared/created but never submitted or enclosed. "
                        "Please add a 'Physical_Combine_or_Package' (enclose) or 'Physical_Submit_Via_Mail' edge to use this item."
                    )

            # --- ケース3: 状態 (State) ---
            elif node_type == NodeType.Process_State:
                continue  # 正常な終了

            # --- ケース4: その他 (Raw_Materialなど) ---
            else:
                self.warnings.append(
                    f"Connectivity Warning: Node '{node['label']}' ({sink_id}) [{node_type}] ends abruptly. "
                    "Typically, Raw_Material or System nodes should not be sinks."
                )

    def _check_cycles(self):
        """循環参照のチェック"""
        try:
            cycle = nx.find_cycle(self.G, orientation="original")
            self.errors.append(
                f"Structure Error: A cycle was detected in the graph {cycle}. "
                "Dependency graphs must be Directed Acyclic Graphs (DAG). Please ensure the flow moves forward in time."
            )
        except nx.NetworkXNoCycle:
            pass

    def _check_type_constraints(self):
        """エッジの型制約チェック（Schemaで定義したルール）"""

        for i, e in enumerate(self.edges):
            src_node = self.node_map[e["source_id"]]
            tgt_node = self.node_map[e["target_id"]]

            src_type = src_node["type"]
            tgt_type = tgt_node["type"]
            edge_type = e["type"]  # Enum値が入っている前提

            # Enumのメンバーとして存在するか確認（念の為）
            # Pydanticでバリデーション済みなら不要だが、安全策として
            try:
                # 文字列からEnumを取得 (definitions.pyのキーと一致するか)
                # self.rulesのキーはEnumオブジェクトそのもの
                current_rule_key = EdgeType(edge_type)
            except ValueError:
                self.errors.append(f"Schema Error: Undefined Edge Type '{edge_type}' found at Edge[{i}].")
                continue

            if current_rule_key in self.rules:
                valid_src, valid_tgt = self.rules[current_rule_key]

                # Sourceチェック
                # valid_srcは Enum または List[Enum]
                if isinstance(valid_src, list):
                    if src_type not in valid_src:
                        valid_names = [v.value for v in valid_src]
                        self.errors.append(
                            f"Type Constraint Error at Edge[{i}] ({edge_type}): "
                            f"Source node '{src_node['label']}' has type '{src_type}', but this edge requires one of {valid_names}."
                        )
                elif src_type != valid_src:
                    self.errors.append(
                        f"Type Constraint Error at Edge[{i}] ({edge_type}): "
                        f"Source node '{src_node['label']}' has type '{src_type}', but this edge requires '{valid_src.value}'."
                    )

                # Targetチェック
                if isinstance(valid_tgt, list):
                    if tgt_type not in valid_tgt:
                        valid_names = [v.value for v in valid_tgt]
                        self.errors.append(
                            f"Type Constraint Error at Edge[{i}] ({edge_type}): "
                            f"Target node '{tgt_node['label']}' has type '{tgt_type}', but this edge requires one of {valid_names}."
                        )
                elif tgt_type != valid_tgt:
                    self.errors.append(
                        f"Type Constraint Error at Edge[{i}] ({edge_type}): "
                        f"Target node '{tgt_node['label']}' has type '{tgt_type}', but this edge requires '{valid_tgt.value}'."
                    )

    def _check_domain_logic(self):
        """ドメイン特有の論理チェック"""

        # 1. アナログ申請の「無からの発生」チェック
        # Physical_Write_or_Process (記入) があるのに、その親に Print や Obtain がない
        fill_edges = [e for e in self.edges if e["type"] == EdgeType.Physical_Write_or_Process]
        for fill in fill_edges:
            paper_id = fill["source_id"]
            # その紙(Raw_Material/Artifact)を作ったエッジを探す
            predecessors = list(self.G.predecessors(paper_id))
            if not predecessors:
                paper_label = self.node_map[paper_id]["label"]
                self.warnings.append(
                    f"Logical Gap: The document '{paper_label}' is being filled/processed, but its origin is unknown. "
                    "Please add a preceding edge like 'Physical_Obtain_Raw_Material', 'Physical_Print_From_Digital', or 'Physical_Possess_PreExistent'."
                )

        # 2. デジタル申請の「多重送信」チェック
        # Digital_Submit_Data が複数ある場合、統合されていない可能性がある
        submit_edges = [e for e in self.edges if e["type"] == EdgeType.Digital_Submit_Data]
        if len(submit_edges) > 1:
            self.warnings.append(
                f"Data Aggregation Warning: {len(submit_edges)} 'Digital_Submit_Data' edges detected. "
                "Usually, application data should be aggregated into a single object before submission. "
                "Check if you can merge data nodes or use 'System_Auto_Link_Data' effectively."
            )

        # 3. Evidenceの欠落チェック
        for i, e in enumerate(self.edges):
            if not e.get("evidence_text") or e.get("evidence_text") == "":
                self.warnings.append(
                    f"Evidence Missing: Edge[{i}] ('{e.get('action', 'unknown')}') lacks 'evidence_text'. "
                    "Please provide a quote from the text or set it to 'implicit'."
                )
