import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from config import TARGET_CITIES
from procedure_analyzer import ProcedureAnalyzer

# .envファイルの読み込み
load_dotenv()


def run_analysis(city_id: str, procedure_name: str = "児童手当 認定請求", output_dir: str = "output/graphs"):
    """
    指定された自治体IDと手続き名で分析を実行し、結果をJSONファイルとして保存する。
    """
    # 自治体情報の取得
    city_info = next((c for c in TARGET_CITIES if c["id"] == city_id), None)
    if not city_info:
        raise ValueError(f"City ID {city_id} not found in configuration.")

    city_name = city_info["name"]
    print(f"Starting analysis for {city_name} ({city_id}) - {procedure_name}...")

    # 出力ディレクトリの作成
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Analyzerの初期化
    analyzer = ProcedureAnalyzer()

    # 実行
    try:
        final_state = analyzer.run(city_name, procedure_name)

        result = final_state.get("analysis_result")
        if result:
            # ファイル保存
            filename = f"{city_id}_{procedure_name}.json"
            file_path = out_path / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"Analysis completed successfully. Saved to {file_path}")
            return str(file_path)
        else:
            print("Analysis failed. No final output generated.")
            return None

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Administrative Procedure Cost Analysis CLI")
    parser.add_argument("city_id", help="Target Municipality ID (e.g., 13105)")
    parser.add_argument("--procedure", default="児童手当 認定請求", help="Target Procedure Name (default: 児童手当 認定請求)")
    parser.add_argument("--output-dir", default="output/graphs", help="Directory to save results")

    args = parser.parse_args()

    try:
        run_analysis(args.city_id, args.procedure, args.output_dir)
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
