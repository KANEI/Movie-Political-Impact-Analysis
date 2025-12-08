import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def count_news(news_str):
    """
    ニュースリストの文字列を解析して件数を返す
    """
    try:
        titles = ast.literal_eval(news_str)
        if isinstance(titles, list):
            return len(titles)
        return 0
    except:
        return 0

def extract_genre_names(x):
    """
    ジャンルの名前をリスト化
    """
    try:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        if isinstance(x, list):
            return [d['name'] for d in x if 'name' in d]
        return []
    except:
        return []

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # --- ニュースの件数をカウント ---
    df['political_news_count'] = df['news'].apply(count_news)
    upper_count = len(df[df['political_news_count']==100])
    upper_rate = (upper_count / len(df)) * 100 if len(df)>0 else 'error'
    print(f"取得上限に達した映画は{upper_count}件で、全体の{upper_rate:.2f}%です")

    # --- 数値変換 ---
    df['budget_log'] = np.log1p(df['budget'])
    df['revenue_log'] = np.log1p(df['revenue'])
    
    # ROI計算
    df['roi'] = (df['revenue'] - df['budget']) / df['budget']
    df['roi_log'] = np.log1p(df['roi'])
    
    # 中央値で高ROIフラグ
    df['is_high_roi'] = (df['roi'] > df['roi'].median()).astype(int)

    # --- 有名度(Fame)の処理 ---
    # actor_fame が 0 の場合もあるので log1p を使用
    df['actor_fame'] = pd.to_numeric(df['actor_fame'], errors='coerce').fillna(0)
    df['actor_fame_log'] = np.log1p(df['actor_fame'])

    # --- ジャンル処理 ---
    df['genre_list'] = df['genres'].apply(extract_genre_names)
    mlb = MultiLabelBinarizer()
    genre_dummies = mlb.fit_transform(df['genre_list'])
    genre_cols = [f"Genre_{c.replace(' ', '_')}" for c in mlb.classes_]
    genre_df = pd.DataFrame(genre_dummies, columns=genre_cols, index=df.index)
    
    # 上位ジャンルのみ
    top_genres = genre_df.sum().sort_values(ascending=False).head(10).index
    df = pd.concat([df, genre_df[top_genres]], axis=1)

    # コレクション
    df["belongs_to_collection"] = df["belongs_to_collection"].notna().astype(int)

    return df