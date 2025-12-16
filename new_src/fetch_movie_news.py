import time
import ast
import pandas as pd
import numpy as np
import kagglehub
from pygooglenews import GoogleNews
from tqdm import tqdm
from datetime import timedelta

# 設定
DATASET_NAME = "rounakbanik/the-movies-dataset"
OUTPUT_FILE = "new_data/movies_with_news.csv"

# フィルタリング条件
FILTER_START_DATE = '2008-01-01'
FILTER_END_DATE = '2016-12-31'
TARGET_COUNTRY = "United States of America"

# ニュース取得設定
NEWS_LANG = 'en'
NEWS_COUNTRY = 'US'
SLEEP_TIME = 1.0

# 政治的キーワード
POLITICAL_KEYWORDS = [
    "politics", "political", "president", "election", "campaign", "vote",
    "democrat", "republican", "white house", "congress", "senate", "policy",
    "government", "activist", "protest", "scandal", "law", "rights",
    "obama", "trump", "bush", "clinton", "biden"
]

def load_dataset():
    print("KaggleHubからデータをダウンロード...")
    path = kagglehub.dataset_download(DATASET_NAME)
    
    print("CSV読み込み...")
    # low_memory=Falseで警告抑制
    metadata = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(f"{path}/credits.csv")
    
    # IDを文字列型にしてマージ
    metadata['id'] = metadata['id'].astype(str)
    credits['id'] = credits['id'].astype(str)
    
    # 必要なカラムだけ残してマージ（メモリ節約）
    meta_cols = ['id', 'title', 'release_date', 'revenue', 'budget', 'production_countries', 'belongs_to_collection', 'genres']
    df = pd.merge(metadata[meta_cols], credits[['id', 'cast']], on='id', how='left')

    # IDの重複削除
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    # インデックスのリセット
    df = df.reset_index(drop=True)
    
    return df

def calculate_star_power(df_all):
    """
    IDの重複を削除したデータセット全体を使って、各映画の「主要俳優の過去3年間の平均興行収入」を計算する。
    （アメリカ以外で製作され、2008年より前に公開された映画も参照する）
    """
    print("俳優の過去実績を計算中...")

    # データクリーニング
    # revenueとdateが有効なもの
    df_valid = df_all.dropna(subset=['release_date', 'revenue', 'cast']).copy()
    df_valid['revenue'] = pd.to_numeric(df_valid['revenue'], errors='coerce')
    df_valid['release_date'] = pd.to_datetime(df_valid['release_date'], errors='coerce')
    df_valid = df_valid[(df_valid['revenue'] > 0) & (df_valid['release_date'].notna())]
    
    # 俳優ごとの出演履歴辞書を作成
    actor_history = {}
    
    print("  出演履歴を構築中...")
    # tqdmで進捗表示
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
        r_date = row['release_date']
        r_rev = row['revenue']
        try:
            cast_list = ast.literal_eval(row['cast'])
            # 上位3名のみを履歴に追加
            top_cast = sorted(cast_list, key=lambda x: x.get('order', 999))[:3]
            for actor in top_cast:
                name = actor['name']
                if name not in actor_history:
                    actor_history[name] = []
                actor_history[name].append((r_date, r_rev))
        except:
            continue

    # 履歴を日付順にソート
    for name in actor_history:
        actor_history[name].sort(key=lambda x: x[0])

    # 各映画について、その公開日時点での過去3年の平均を計算
    fame_scores = []
    
    print("  各映画の俳優の有名度を算出中...")
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all)):
        current_date = pd.to_datetime(row['release_date'], errors='coerce')
        if pd.isna(current_date):
            fame_scores.append(0)
            continue
            
        start_window = current_date - timedelta(days=365*3)
        
        movie_actor_revenues = []
        
        try:
            cast_list = ast.literal_eval(row['cast'])
            top_cast = sorted(cast_list, key=lambda x: x.get('order', 999))[:3]
            
            for actor in top_cast:
                name = actor['name']
                history = actor_history.get(name, [])
                
                # 公開日より前、かつ3年以内の期間内の映画を抽出
                recent_movies = [rev for date, rev in history if start_window <= date < current_date]
                
                if recent_movies:
                    avg_rev = sum(recent_movies) / len(recent_movies)
                    movie_actor_revenues.append(avg_rev)
                else:
                    movie_actor_revenues.append(0)
        except:
            pass
            
        # 3人の平均をその映画のFameスコアとする
        if movie_actor_revenues:
            fame_scores.append(sum(movie_actor_revenues) / len(movie_actor_revenues))
        else:
            fame_scores.append(0)
            
    df_all['actor_fame'] = fame_scores
    return df_all

def preprocess_target_data(df):
    """
    分析対象（2008-2016, US）のフィルタリング
    """
    print("分析対象データのフィルタリング...")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    
    mask = (df['release_date'] >= pd.to_datetime(FILTER_START_DATE)) & \
           (df['release_date'] <= pd.to_datetime(FILTER_END_DATE))
    df = df[mask]
    
    # 国フィルタ
    df = df[df['production_countries'].fillna("").str.contains(TARGET_COUNTRY)]
    
    # 予算・収入の数値化
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    
    return df.reset_index(drop=True)

def create_political_query(row):
    """
    (Actor1 OR Actor2) AND (politics OR election ...) 形式のクエリを作成
    """
    if pd.isnull(row['release_date']):
        return None

    # 期間: 3ヶ月前から2週間後
    start = row['release_date'] - pd.DateOffset(months=3)
    end = row['release_date'] + pd.DateOffset(weeks=2)
    
    # 俳優クエリ
    actors_str = ""
    try:
        cast_list = ast.literal_eval(row['cast'])
        top_cast = sorted(cast_list, key=lambda x: x.get('order', 999))[:3]
        names = [f'"{c["name"]}"' for c in top_cast if "name" in c]
        if names:
            actors_str = "(" + " OR ".join(names) + ")"
    except:
        return None

    if not actors_str:
        return None

    # 政治キーワードクエリ
    politics_str = "(" + " OR ".join(POLITICAL_KEYWORDS) + ")"
    
    # 結合
    query = f"{actors_str} AND {politics_str} after:{start.strftime('%Y-%m-%d')} before:{end.strftime('%Y-%m-%d')}"
    return query

def main():
    # 全データロード
    df_raw = load_dataset()
    
    # 有名度計算
    # フィルタリング前に計算しないと、履歴データが不足するため
    df_with_fame = calculate_star_power(df_raw)
    
    # 分析対象の抽出
    df = preprocess_target_data(df_with_fame)
    print(f"分析対象データ数: {len(df)}件")

    # クエリ生成
    print("検索クエリ生成...")
    df['query'] = df.apply(create_political_query, axis=1)
    df = df.dropna(subset=['query'])

    # ニュース取得
    print("ニュース取得開始...")
    gn = GoogleNews(lang=NEWS_LANG, country=NEWS_COUNTRY)
    
    news_results = []
    # テスト用に最初の5件だけ表示する等の制限は外していますが、
    # 実行時間が長くなる場合は適宜調整してください。
    for q in tqdm(df['query']):
        try:
            res = gn.search(q)
            titles = [entry['title'] for entry in res.get('entries', [])]
            news_results.append(titles)
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"Error: {e}")
            news_results.append([])
            
    df['news'] = news_results
    
    # 保存
    print(f"保存中: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("完了")

if __name__ == "__main__":
    main()