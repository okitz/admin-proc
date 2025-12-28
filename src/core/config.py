# LLM Models
MODEL_FAST = "gemini-2.5-flash"
MODEL_HIGH_QUALITY = "gemini-2.5-pro"
# MODEL_FAST = "gemini-2.5-flash-lite"
# MODEL_HIGH_QUALITY = "gemini-2.5-flash"

# Search
DEFAULT_SEARCH_K = 10

# Control Flow Limits
MAX_SEARCH_LOOPS = 5
MAX_GENERATION_RETRIES = 10

# 対象自治体
TARGET_CITIES = [
    {"id": "01100", "name": "札幌市", "prefecture": "北海道"},
    {"id": "04100", "name": "仙台市", "prefecture": "宮城県"},
    {"id": "13105", "name": "文京区", "prefecture": "東京都"},
    {"id": "13112", "name": "世田谷区", "prefecture": "東京都"},
    {"id": "13103", "name": "港区", "prefecture": "東京都"},
    {"id": "13113", "name": "渋谷区", "prefecture": "東京都"},
    {"id": "13121", "name": "足立区", "prefecture": "東京都"},
    {"id": "23100", "name": "名古屋市", "prefecture": "愛知県"},
    {"id": "27100", "name": "大阪市", "prefecture": "大阪府"},
    {"id": "34100", "name": "広島市", "prefecture": "広島県"},
    {"id": "40130", "name": "福岡市", "prefecture": "福岡県"},
]
