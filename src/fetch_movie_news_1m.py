import time
import ast
from typing import Optional, List

import kagglehub
import pandas as pd
from pygooglenews import GoogleNews
from tqdm import tqdm

# 設定・定数
DATASET_NAME = "rounakbanik/the-movies-dataset"
OUTPUT_FILE = "data/movies_with_news_1m.csv"

# フィルタリング条件
FILTER_START_DATE = '2008-01-01'
FILTER_END_DATE = '2016-12-31'
TARGET_COUNTRY = "United States of America"

# ニュース取得設定
NEWS_LANG = 'en'
NEWS_COUNTRY = 'US'
SLEEP_TIME = 1.0  # リクエスト間隔(秒)

def load_dataset() -> pd.DataFrame:
    """
    Kaggleからデータセットをダウンロードし、マージして返す関数
    """
    print("KaggleHubからデータをダウンロード...")
    path = kagglehub.dataset_download(DATASET_NAME)
    
    print("CSVファイルの読み込み...")
    metadata = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    keywords = pd.read_csv(f"{path}/keywords.csv")
    credits = pd.read_csv(f"{path}/credits.csv")

    # IDを文字列型に統一（マージキー）
    metadata['id'] = metadata['id'].astype(str)
    keywords['id'] = keywords['id'].astype(str)
    credits['id'] = credits['id'].astype(str)

    print("データフレームを結合...")
    merged = pd.merge(metadata, keywords, on="id", how="left")
    merged = pd.merge(merged, credits, on="id", how="left")
    
    return merged

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データのクリーニングとフィルタリングを行う関数
    """
    print("データ加工...")
    
    # 必要なカラムが欠損している行を削除
    df = df.dropna(subset=['release_date', 'revenue', 'cast'])
    
    # 予算と収入が0のデータを除外
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df = df[(df['revenue'] > 0) & (df['budget'] > 0)]

    # 日付型へ変換
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])

    # 日付でフィルタリング
    mask_date = (df['release_date'] >= pd.to_datetime(FILTER_START_DATE)) & \
                (df['release_date'] <= pd.to_datetime(FILTER_END_DATE))
    df = df[mask_date]

    # 国でフィルタリング
    df = df[df['production_countries'].fillna("").str.contains(TARGET_COUNTRY)]

    # IDの重複削除
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    # インデックスのリセット
    df = df.reset_index(drop=True)
    
    print(f"データの加工完了. 行数は {len(df)}")
    return df

def create_search_query(row: pd.Series) -> Optional[str]:
    """
    各行に対してGoogle News検索用のクエリを生成する関数
    形式: (Actor1 OR Actor2 OR Actor3) after:YYYY-MM-DD before:YYYY-MM-DD
    """
    if pd.isnull(row['release_date']):
        return None

    # 期間設定: 公開1ヶ月前から公開2週間後まで
    start_date = row['release_date'] - pd.DateOffset(months=1)
    end_date = row['release_date'] + pd.DateOffset(weeks=2)
    
    after_str = start_date.strftime('%Y-%m-%d')
    before_str = end_date.strftime('%Y-%m-%d')

    actors_query = ""
    try:
        # 文字列形式のリストをPythonオブジェクトへ変換
        cast_list = ast.literal_eval(row['cast'])
        
        if isinstance(cast_list, list):
            # order順にソートして上位3名を取得
            top_cast = sorted(cast_list, key=lambda x: x.get('order', 999))[:3]
            
            # 名前を抽出してダブルクォートで囲む
            names = [f'"{c["name"]}"' for c in top_cast if "name" in c]
            
            # OR で結合
            if names:
                actors_query = " OR ".join(names)
    except (ValueError, SyntaxError, TypeError):
        pass

    if not actors_query:
        return None

    return f'({actors_query}) after:{after_str} before:{before_str}'

def fetch_news_titles(gn_client: GoogleNews, query_str: str) -> List[str]:
    """
    GoogleNewsクライアントを使用してタイトルを取得する関数
    """
    if not query_str:
        return []

    try:
        search_result = gn_client.search(query_str)
        titles = [entry['title'] for entry in search_result.get('entries', [])]
        time.sleep(SLEEP_TIME) # レートリミット対策
        return titles
    except Exception as e:
        print(f"Error searching query: {query_str} | Error: {e}")
        return []

def main():
    # データロード
    raw_df = load_dataset()
    
    # 前処理
    df = preprocess_data(raw_df)

    # クエリ生成
    print("クエリを生成...")
    df['query'] = df.apply(create_search_query, axis=1)
    
    # クエリ生成に失敗した行を除外する
    df = df.dropna(subset=['query'])

    # ニュース取得
    print("ニュースタイトルを取得...")
    gn = GoogleNews(lang=NEWS_LANG, country=NEWS_COUNTRY)
    
    # tqdmを使って進捗を表示しながら適用
    news_results = []
    for query in tqdm(df['query'], desc="Fetching News"):
        titles = fetch_news_titles(gn, query)
        news_results.append(titles)
    
    df['news'] = news_results

    # 保存
    print(f"次のCSVファイルとして保存： {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("完了！")

if __name__ == "__main__":
    main()