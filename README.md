# 俳優の政治的発言が映画の成功を左右するか
ハリウッド俳優の政治的発言がアメリカ映画の興行収入・ROIに与える影響の分析

## 概要
このプロジェクトは、映画公開前後にキャストに関する「政治的なニュース」がどの程度報道されたかを定量化し、それが映画の興行収入 (Revenue) や 投資対効果 (ROI)に与える影響を統計的に分析します。
Kaggleの映画データセットとGoogle Newsの検索結果を結合し、頻度論的アプローチ（Statsmodels）とベイズ統計的アプローチ（PyMC）の両方を用いてモデリングを行います。

## ディレクトリ構成

```
.
├── .venv/                      # 仮想環境
├── data/                       # データセット保存用
├── image/                      # 分析結果のグラフ保存用
├── notebooks/                  # jupyter notebook
├── src/                        # ソースコード格納ディレクトリ
│   ├── fetch_movie_news_1m.py      # ニュース取得 (公開前後1ヶ月)
│   ├── fetch_movie_news_3m.py      # ニュース取得 (公開前3ヶ月〜後2週間)
│   ├── analyze_political_news_1m.py # ニュース記事の分析・データ加工 (1mデータ用)
│   ├── analyze_political_news_3m.py # ニュース記事の分析・データ加工(3mデータ用)
│   ├── data_processing.py          # 前処理・特徴量エンジニアリング用モジュール
│   ├── visualization.py            # 可視化・グラフ描画用モジュール
│   ├── main.py                     # 頻度論的統計分析 (OLS, ロジスティック回帰, 感度分析)
│   └── bayesian_analysis.py        # ベイズ統計分析 (MCMC, PyMC)
├── .gitignore                  # githubにアップロードしないファイル
├── requirements.txt            # 必要なパッケージ
└── README.md                   # 本ファイル
```

## 必要要件

Python 3.8 以上。必要なパッケージはrequirements.txtに記載してあります。

## 環境構築と実行手順

分析は以下の3ステップで行います。期間設定（1ヶ月 or 3ヶ月）に応じてスクリプトを選択してください。

### 0\. 環境構築
プロジェクトのルートディレクトリで以下のコマンドを実行し、仮想環境を作成して依存パッケージをインストールしてください。
```bash
# 仮想環境 (.venv) の作成
python -m venv .venv

# 仮想環境の有効化
# Windowsの場合:
.venv\Scripts\activate
# Mac/Linuxの場合:
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 1\. データの取得

Kaggleから映画データをダウンロードし、Google Newsからキャストに関するニュースタイトルを取得します。

**1ヶ月版を実行する場合:**

```bash
python fetch_movie_news_1m.py
```

**3ヶ月版を実行する場合:**

```bash
python fetch_movie_news_3m.py
```

*出力: `data/movies_with_news_xm.csv`*

### 2\. ニュース記事の分析・データ加工

取得したニュースタイトルに含まれる政治的キーワード（"election", "vote", "protest"など）をカウントし、政治色の強さを数値化し、統合したデータを作成します。

**1ヶ月版:**

```bash
python analyze_political_news_1m.py
```

**3ヶ月版:**

```bash
python analyze_political_news_3m.py
```

*出力: `data/movies_analyzed_xm.csv`*

### 3\. 統計モデリング

**A. 頻度論的分析 (Frequentist Analysis)**
Statsmodelsを用いたOLS（最小二乗法）およびロジスティック回帰を実行します。閾値の感度分析も含まれます。
※ `main.py` 内の `INPUT_FILE` 変数で使用するCSVを指定してください（デフォルトは `3m`）。

```bash
python main.py
```

*出力: 回帰分析のサマリ、分布図、感度分析プロットなどが `image/` フォルダに保存されます。*

**B. ベイズ統計分析 (Bayesian Analysis)**
PyMCを用いたMCMCサンプリングを行い、事後分布の推定を行います。特に「政治的ニュースの割合」と「ニュース総数」の交互作用項に注目します。
※ `bayesian_analysis.py` 内の `INPUT_FILE` 変数で使用するCSVを指定してください。

```bash
python bayesian_analysis.py
```

*出力: トレースプロット、事後分布プロットなどが `image/` フォルダに保存されます。*

## 分析内容の詳細
ニュースの件数には、1映画あたり最大100件という取得上限が存在した。収集されたデータの内訳は、上限（100件）に達した映画が948件、上限に満たない映画が464件であった。前者は母集団（その映画に関する全ニュース）からの「100件の標本」に基づく割合であるのに対し、後者は「全数」に基づく割合であるため、両者のサンプリングの性質は異なる。
したがって本調査では、この構造的な違いを制御するために上限到達ダミー変数（over_100news）をモデルに投入し、政治的ニュース割合(political_ratio)との交互作用を検証する分析手法をとった。\par
また、頻度論の検定は帰無仮説を棄却しないとそこまで説得力を持たないため、事前分布を設定し、ベイズ推定で係数の事後分布を推定する。

### 1\. 特徴量エンジニアリング (`data_processing.py`)

  * **目的変数:**
      * `revenue_log`: 興行収入の対数
      * `roi_log`: ROI (Revenue/Budget) の対数
      * `is_high_roi`: ROIが中央値を超えているか (Binary)
  * **説明変数:**
      * `political_ratio`: ニュースタイトル中の政治的単語の割合
      * `over_100news`: ニュース件数が上限(100件)に達しているか
      * `budget_log`: 予算の対数
      * `belong_to_collection`: シリーズ作品かどうか
      * `Genre`: ジャンル（Top10）

### 2\. 可視化 (`visualization.py`)
  * ヒストグラム（予算、収入、ニュース数）
  * Q-Qプロット
  * ヒートマップ（クロス集計）
  * 回帰分析結果のサマリ画像出力
  * ベイズモデルの事後分布の画像出力

### 3\. モデリング
  * **Model 1 & 2 (OLS):** `revenue_log` および `roi_log` に対する線形回帰。
  * **Model 3 (Logistic):** `is_high_roi` に対するロジスティック回帰。
  * **感度分析:** `is_high_roi` を決定する閾値を変動させた際、説明変数の係数がどう変化するかを確認。