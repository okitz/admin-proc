import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from agent.analyzer import ProcedureAnalyzer
from core.config import TARGET_CITIES

# .envファイルの読み込み
load_dotenv()


class AnalysisRunner:
    """
    手続き分析を実行し、結果を管理するクラス。
    """

    def __init__(self, procedure_name: str = "児童手当 認定請求", output_dir: str = "output/graphs"):
        """
        AnalysisRunnerを初期化します。

        Args:
            procedure_name (str): 分析対象の手続き名。
            output_dir (str): 結果を保存するディレクトリ。
        """
        self.procedure_name = procedure_name
        self.output_dir = Path(output_dir)
        self.analyzer = ProcedureAnalyzer()

        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(self, city_id: str) -> str | None:
        """
        指定された単一の自治体IDで分析を実行し、結果をJSONファイルとして保存する。

        Args:
            city_id (str): 分析対象の自治体ID。

        Returns:
            Optional[str]: 保存されたファイルパス。失敗した場合はNone。
        """
        city_info = next((c for c in TARGET_CITIES if c["id"] == city_id), None)
        if not city_info:
            print(f"Warning: City ID {city_id} not found in configuration. Skipping.")
            return None

        city_name = city_info["name"]
        print(f"Starting analysis for {city_name} ({city_id}) - {self.procedure_name}...")

        # ファイルパスの生成
        filename = f"{city_id}_{self.procedure_name}.json"
        file_path = self.output_dir / filename

        # ファイルの存在チェック
        if file_path.exists():
            print(f"File {file_path} already exists. Skipping.")
            return str(file_path)

        try:
            config: RunnableConfig = {"recursion_limit": 50}  # デフォルトの25から50に引き上げ
            final_state = self.analyzer.run(city_name, self.procedure_name, config=config)
            result = final_state.get("analysis_result")
            if result:
                # ファイル保存
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                print(f"Analysis completed successfully. Saved to {file_path}")
                return str(file_path)

            print("Analysis failed. No final output generated.")
            return None
        except Exception as e:
            print(f"An error occurred during execution for {city_name}: {e}")
            raise e

    def run_for_all_targets(self) -> dict[str, str | None]:
        """設定されているすべての対象自治体に対して分析を実行する。"""
        results = {}
        for city in TARGET_CITIES:
            city_id = city["id"]
            try:
                results[city_id] = self.run_single(city_id)
            except Exception as e:
                print(f"An error occurred during analysis for city {city_id}: {e}")
                results[city_id] = None
            print("-" * 50)
        return results
