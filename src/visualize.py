"""
グラフデータの可視化・分析スクリプト

output/graphsからJSONファイルを読み込み、NetworkXでグラフを可視化し、
コスト分析（積み上げ棒グラフ等）を実行する。
"""
import os
import json
import glob
import argparse
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib as mpl
from matplotlib import font_manager
import pandas as pd
import numpy as np

import config

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']

# コストテーブル（単位: 分）
COST_TABLE = {
    # Physical Actions
    "Physical_Acquire_Resident": 15,  # 役所で取得
    "External_Acquire": 60,           # 外部機関からの取得
    "Physical_Print_Store": 5,        # 店舗で印刷
    "Physical_Print_Home": 5,         # 自宅で印刷
    "Physical_Get_Material": 2,       # 材料取得
    "Physical_Fill": 15,              # 手書き記入
    "Physical_Copy_Store": 5,         # コピー作業
    "Physical_Enclose": 2,            # 書類添付・整理
    "Physical_Submit_Window": 15,     # 窓口提出（移動含む）
    "Physical_Submit_Mail": 10,       # 郵送作業
    
    # Digital Actions
    "Digital_Access": 1,              # アクセス
    "Digital_Download": 2,            # ダウンロード
    "Digital_Input": 10,              # Web入力
    "Digital_Auth": 5,                # 認証・電子署名
    "Digital_Capture": 5,             # 撮影・スキャン
    "Digital_Upload": 2,              # ファイルアップロード
    "Digital_Submit": 1,              # 送信ボタン
    
    # Time/Absence Actions
    "No_Action": 0,                   # 省略されたアクション
    "Wait_Result": 0,                 # 待機時間（コストとして計上しない）
}

# カテゴリーごとの色分け（可視化用）
TYPE_COLORS = {
    "Physical_Acquire_Resident": "#FF6B6B",
    "External_Acquire": "#9B59B6",
    "Physical_Print_Store": "#FFD1D1",
    "Physical_Print_Home": "#FFB3B3",
    "Physical_Get_Material": "#FFCDB3",
    "Physical_Fill": "#FFA07A",
    "Physical_Copy_Store": "#FF8E8E",
    "Physical_Enclose": "#FFCDB3",
    "Physical_Submit_Window": "#FF6B6B",
    "Physical_Submit_Mail": "#FF8E8E",
    
    "Digital_Access": "#74B9FF",
    "Digital_Download": "#96CEB4",
    "Digital_Input": "#4ECDC4",
    "Digital_Auth": "#45B7D1",
    "Digital_Capture": "#FFEAA7",
    "Digital_Upload": "#96CEB4",
    "Digital_Submit": "#74B9FF",
    
    "No_Action": "#B2BEC3",
    "Wait_Result": "#DFE6E9",
}


def load_graph_data(file_path: str) -> Dict:
    """JSONファイルからグラフデータを読み込む"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_networkx_graph(nodes: List[Dict], edges: List[Dict]) -> nx.DiGraph:
    """ノードとエッジリストからNetworkXの有向グラフを作成"""
    G = nx.DiGraph()
    
    # ノードを追加
    for node in nodes:
        G.add_node(node["id"], label=node["label"], type=node["type"])
    
    # エッジを追加
    for edge in edges:
        G.add_edge(
            edge["source_id"],
            edge["target_id"],
            description=edge["description"],
            type=edge["type"],
            cost=COST_TABLE.get(edge["type"], 0)
        )
    return G


def calculate_total_cost(G: nx.DiGraph) -> float:
    """グラフの総コストを計算"""
    total = 0
    for u, v, data in G.edges(data=True):
        total += data.get("cost", 0)
    return total


def get_cost_breakdown(G: nx.DiGraph) -> Dict[str, float]:
    """タイプ別のコスト内訳を取得"""
    breakdown = {}
    for u, v, data in G.edges(data=True):
        action_type = data.get("type", "Unknown")
        cost = data.get("cost", 0)
        breakdown[action_type] = breakdown.get(action_type, 0) + cost
    return breakdown


def visualize_graph(G: nx.DiGraph, title: str, output_path: str):
    """グラフを可視化して保存"""
    plt.figure(figsize=(14, 10))
    
    # レイアウト計算
    # pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    pos = nx.planar_layout(G, scale=1, center=None, dim=2)
    
    # ノードの描画
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "")
        if "State" in node_type:
            node_colors.append("#FFE5E5")
        elif "System" in node_type:
            node_colors.append("#E5F5FF")
        elif "Digital" in node_type:
            node_colors.append("#E5FFE5")
        else:
            node_colors.append("#F5F5F5")
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2000, alpha=0.9)
    
    # エッジの描画
    nx.draw_networkx_edges(G, pos, edge_color="#888888", 
                          arrows=True, arrowsize=20, width=2)
    
    # ラベルの描画
    labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")
    
    plt.title(title, fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def create_cost_comparison_chart(all_data: Dict, output_path: str):
    """コスト比較の積み上げ棒グラフを作成"""
    cities = list(all_data.keys())
    city_names = [all_data[cid]["name"] for cid in cities]
    
    # データ準備
    analog_costs = [all_data[cid]["analog_cost"] for cid in cities]
    digital_costs = [all_data[cid]["digital_cost"] for cid in cities]
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cities))
    width = 0.35
    
    ax.bar(x - width/2, analog_costs, width, label='アナログ申請', color='#FF6B6B')
    ax.bar(x + width/2, digital_costs, width, label='デジタル申請', color='#4ECDC4')
    
    ax.set_ylabel('所要時間（分）', fontsize=12)
    ax.set_title('自治体別 申請手続きコスト比較', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(city_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def analyze_graphs(input_dir: str, output_dir: str):
    """グラフデータを分析"""
    # JSONファイルを検索
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = {}
    
    for json_file in json_files:
        city_id = os.path.basename(json_file).replace(".json", "").split("_")[0]
        
        print(f"\nProcessing: {city_id}")
        
        # データ読み込み
        graph_data = load_graph_data(json_file)
        
        # グラフ作成
        analog_graph = create_networkx_graph(
            graph_data.get("analog_nodes", []),
            graph_data.get("analog_edges", [])
        )
        digital_graph = create_networkx_graph(
            graph_data.get("digital_nodes", []),
            graph_data.get("digital_edges", [])
        )
        
        # コスト計算
        analog_cost = calculate_total_cost(analog_graph)
        digital_cost = calculate_total_cost(digital_graph)
        reduction_rate = ((analog_cost - digital_cost) / analog_cost * 100) if analog_cost > 0 else 0
        
        print(f"  アナログコスト: {analog_cost}分")
        print(f"  デジタルコスト: {digital_cost}分")
        print(f"  削減率: {reduction_rate:.1f}%")
        
        # データ保存
        all_data[city_id] = {
            "name": city_id,
            "analog_graph": analog_graph,
            "digital_graph": digital_graph,
            "analog_cost": analog_cost,
            "digital_cost": digital_cost,
            "reduction_rate": reduction_rate,
            "analog_breakdown": get_cost_breakdown(analog_graph),
            "digital_breakdown": get_cost_breakdown(digital_graph)
        }
        
        # グラフ可視化
        visualize_graph(
            analog_graph,
            f"{city_id} - アナログ申請フロー",
            os.path.join(output_dir, f"{city_id}_analog_graph.png")
        )
        visualize_graph(
            digital_graph,
            f"{city_id} - デジタル申請フロー",
            os.path.join(output_dir, f"{city_id}_digital_graph.png")
        )
    
    # 比較チャート作成
    if len(all_data) > 1:
        create_cost_comparison_chart(
            all_data,
            os.path.join(output_dir, "cost_comparison.png")
        )
    
    print(f"\n✓ Analysis complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze procedure graphs")
    parser.add_argument("--input-dir", default="output/graphs", 
                       help="Directory containing JSON graph files")
    parser.add_argument("--output-dir", default="output/visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    analyze_graphs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
