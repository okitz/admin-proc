# 大規模言語モデルを用いた行政手続き文書のグラフ構造化とDX効果の定量的評価
https://github.com/okitz/admin-proc

## 実行手順

1. 依存パッケージのインストール
```bash 
uv sync # or pip install -r requirements.txt
```

2. (グラフ生成を行う場合) 環境変数の設定, Configファイルの編集
`.env`ファイルに以下を設定
```
GOOGLE_API_KEY=(Gemini 用のAPIキー)
TAVILY_API_KEY=(Tavily 用のAPIキー)
```

`src/core/config.py` の設定を必要に応じて編集。

3. ノートブックの実行
`src/admin-proc.ipynb`をJupyter Notebook等で開き、セルを実行する。
レポート作成に用いたグラフデータは `output/graphs` に同梱済み。