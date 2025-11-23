import os
from pathlib import Path

from dotenv import load_dotenv

# .envファイルを読み込む
# .envファイルはプロジェクトのルートディレクトリに配置する
load_dotenv()

# APIキー
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# ディレクトリ設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_TEXT_DIR = DATA_DIR / "raw_text"
RAW_HTML_DIR = DATA_DIR / "raw_html"
PROC_GRAPH_DIR = DATA_DIR / "processed_graph"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 確認用
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir:     {DATA_DIR}")
    # print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")
    print(f"CUSTOM_SEARCH_ENGINE_ID: {CUSTOM_SEARCH_ENGINE_ID}")


# 対象自治体
TARGET_CITIES = [
    {
        "id": "01100",
        "name": "札幌市",
        "prefecture": "北海道"
    },
    {
        "id": "04100",
        "name": "仙台市",
        "prefecture": "宮城県"
    },
    {
        "id": "13105",
        "name": "文京区",
        "prefecture": "東京都"
    },
    {
        "id": "13112",
        "name": "世田谷区",
        "prefecture": "東京都"
    },
    {
        "id": "13103",
        "name": "港区",
        "prefecture": "東京都"
    },
    {
        "id": "13113",
        "name": "渋谷区",
        "prefecture": "東京都"
    },
    {
        "id": "13121",
        "name": "足立区",
        "prefecture": "東京都"
    },
    {
        "id": "23100",
        "name": "名古屋市",
        "prefecture": "愛知県"
    },
    {
        "id": "27100",
        "name": "大阪市",
        "prefecture": "大阪府"
    },
    {
        "id": "34100",
        "name": "広島市",
        "prefecture": "広島県"
    },
    {
        "id": "40130",
        "name": "福岡市",
        "prefecture": "福岡県"
    },
]
