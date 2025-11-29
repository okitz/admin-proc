import argparse
import glob
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 定義ファイルからEnumとメタデータをインポート
from definitions import EDGE_DEFINITIONS, EdgeType, NodeType

# 日本語フォントの設定 (環境に合わせて調整してください)
# Linux/Macなら "Noto Sans CJK JP", Windowsなら "Meiryo" や "MS Gothic"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans", "Meiryo"]

# --- 定義ファイルからスタイル辞書を生成 ---
EDGE_STYLES = {edge_enum: meta["meta"] for edge_enum, meta in EDGE_DEFINITIONS.items()}

# ノードの色定義 (definitions.pyにない場合はここで定義)
NODE_COLORS = {
    NodeType.Process_State: "#FFCCCC",  # 赤系 (State)
    NodeType.Digital_System: "#E5F5FF",  # 青系 (System)
    NodeType.Digital_Data_Object: "#E5FFE5",  # 緑系 (Data)
    NodeType.Physical_Raw_Material: "#FFFFCC",  # 黄系 (Raw)
    NodeType.Physical_Processed_Artifact: "#F5F5F5",  # グレー系 (Artifact)
}


def load_graph_data(file_path: str) -> dict[str, Any]:
    """JSONファイルからグラフデータを読み込む"""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def create_networkx_graph(nodes: list[dict], edges: list[dict]) -> nx.DiGraph:
    """ノードとエッジリストからNetworkXの有向グラフを作成"""
    G = nx.DiGraph()

    # ノードを追加
    for node in nodes:
        G.add_node(node["id"], label=node["label"], type=node["type"])

    # エッジを追加
    for edge in edges:
        # 文字列のTypeからEnumを取得 (存在しない場合はデフォルト値を設定)
        try:
            etype = EdgeType(edge["type"])
            meta = EDGE_STYLES.get(etype, {"base_cost": 0, "color": "black", "style": "solid"})
        except ValueError:
            meta = {"base_cost": 0, "color": "black", "style": "solid"}

        G.add_edge(
            edge["source_id"],
            edge["target_id"],
            description=edge.get("description", ""),
            type=edge["type"],
            cost=meta["base_cost"],
            color=meta.get("color", "black"),
            style=meta.get("style", "solid"),
        )
    return G


def calculate_total_cost(G: nx.DiGraph) -> float:
    """グラフの総コストを計算"""
    total = 0
    for _, _, data in G.edges(data=True):
        total += data.get("cost", 0)
    return total


def get_cost_breakdown(G: nx.DiGraph) -> dict[str, float]:
    """タイプ別のコスト内訳を取得"""
    breakdown = {}
    for _, _, data in G.edges(data=True):
        action_type = data.get("type", "Unknown")
        cost = data.get("cost", 0)
        breakdown[action_type] = breakdown.get(action_type, 0) + cost
    return breakdown


def visualize_graph(G: nx.DiGraph, title: str, output_path: str):
    """グラフを可視化して保存"""
    plt.figure(figsize=(14, 10))

    # レイアウト計算
    # 階層構造が見やすいレイアウトを使用 (dotなどがあればベターだが標準ならspringかkamada_kawai)
    try:
        # pygraphvizが必要な場合があるため、なければspring_layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except ImportError:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # ノード色の決定
    node_colors = []
    for node in G.nodes():
        node_type_str = G.nodes[node].get("type", "")
        # Enum文字列と一致する色を取得
        color = "#FFFFFF"  # Default
        for ntype, c in NODE_COLORS.items():
            if ntype.value == node_type_str:
                color = c
                break
        node_colors.append(color)

    # ノードの描画
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, edgecolors="gray", alpha=0.9)

    # エッジの描画 (色やスタイルを個別に適用)
    edges = G.edges(data=True)
    for u, v, data in edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=data.get("color", "black"),
            style=data.get("style", "solid"),
            arrows=True,
            arrowsize=20,
            width=2,
        )

    # ラベルの描画 (改行を入れる等の工夫も可能)
    labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family="sans-serif")

    # エッジラベル (コストを表示)
    edge_labels = {(u, v): f"{d.get('cost')}min" for u, v, d in edges if d.get("cost", 0) > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title, fontsize=16, pad=20)
    plt.axis("off")

    # 凡例の追加 (簡易的)
    # plt.legend(...) # 必要なら実装

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved graph image: {output_path}")


def create_cost_comparison_chart(all_data: dict, output_path: str):
    """コスト比較の積み上げ棒グラフを作成"""
    cities = list(all_data.keys())
    # ファイル名から自治体名を抽出（文京区_2024... -> 文京区）
    city_names = [all_data[cid]["name"] for cid in cities]

    # データ準備
    analog_costs = [all_data[cid]["analog_cost"] for cid in cities]
    digital_costs = [all_data[cid]["digital_cost"] for cid in cities]

    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cities))
    width = 0.35

    rects1 = ax.bar(x - width / 2, analog_costs, width, label="アナログ申請", color="#FF6B6B")
    rects2 = ax.bar(x + width / 2, digital_costs, width, label="デジタル申請", color="#4ECDC4")

    # 値のラベル表示
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    ax.set_ylabel("推定所要時間（分）", fontsize=12)
    ax.set_title("自治体別 申請手続き摩擦コスト比較", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(city_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart: {output_path}")


def analyze_graphs(input_dir: str, output_dir: str):
    """ディレクトリ内の全JSONファイルを分析"""
    # JSONファイルを検索
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    all_data = {}

    for json_file in json_files:
        # ファイル名からID/自治体名を取得
        filename = os.path.basename(json_file)
        city_id = filename.split("_")[0]  # "文京区_timestamp.json" -> "文京区"

        print(f"\nProcessing: {city_id}")

        try:
            # データ読み込み
            graph_data = load_graph_data(json_file)

            # analysis_result直下にあるか、final_output内にあるかで分岐（Schemaの変更に対応）
            if "analysis_result" in graph_data:
                res = graph_data["analysis_result"]
            elif "final_graph" in graph_data:  # 旧スキーマ対応
                res = graph_data["final_graph"]
            else:
                res = graph_data  # フラットな場合

            # グラフ作成
            analog_graph = create_networkx_graph(res.get("analog_nodes", []), res.get("analog_edges", []))
            digital_graph = create_networkx_graph(res.get("digital_nodes", []), res.get("digital_edges", []))

            # コスト計算
            analog_cost = calculate_total_cost(analog_graph)
            digital_cost = calculate_total_cost(digital_graph)

            if analog_cost > 0:
                reduction_rate = (analog_cost - digital_cost) / analog_cost * 100
            else:
                reduction_rate = 0

            print(f"  アナログコスト: {analog_cost}分")
            print(f"  デジタルコスト: {digital_cost}分")
            print(f"  削減率: {reduction_rate:.1f}%")

            # データ保存
            all_data[city_id] = {
                "name": city_id,
                "analog_cost": analog_cost,
                "digital_cost": digital_cost,
                "reduction_rate": reduction_rate,
            }

            # グラフ可視化 (個別に保存)
            visualize_graph(
                analog_graph,
                f"{city_id} - アナログ申請フロー (コスト: {analog_cost}分)",
                os.path.join(output_dir, f"{filename}_analog.png"),
            )
            visualize_graph(
                digital_graph,
                f"{city_id} - デジタル申請フロー (コスト: {digital_cost}分)",
                os.path.join(output_dir, f"{filename}_digital.png"),
            )

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback

            traceback.print_exc()

    # 比較チャート作成 (全自治体分)
    if len(all_data) > 0:
        create_cost_comparison_chart(all_data, os.path.join(output_dir, "total_comparison.png"))

    print(f"\n✓ Analysis complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze procedure graphs")
    parser.add_argument("--input-dir", default="./output/graphs", help="Directory containing JSON graph files")
    parser.add_argument("--output-dir", default="./output/viz", help="Directory to save visualizations")

    args = parser.parse_args()

    analyze_graphs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
