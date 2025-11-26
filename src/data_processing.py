import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def extract_genre_names(x):
    """
    文字列 "[{'id': 35, 'name': 'Comedy'}, ...]" をリスト ['Comedy', 'Romance'] に変換
    """
    try:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        if isinstance(x, list):
            return [d['name'] for d in x if 'name' in d]
        return []
    except (ValueError, SyntaxError):
        return []

def load_and_preprocess_data(filepath):
    """
    データの読み込みから特徴量エンジニアリングまでを一括で行う
    """
    df = pd.read_csv(filepath)

    # --- 数値変換 (対数) ---
    # log1p は log(x + 1)
    df['budget_log'] = np.log1p(df['budget'])
    df['revenue_log'] = np.log1p(df['revenue'])

    # --- ROIの計算 ---
    # ROI = (収益 - 予算) / 予算
    df['roi'] = (df['revenue'] - df['budget']) / df['budget']
    df['roi_log'] = np.log1p(df['roi']) # 注意: ROIが-1以下だとNaNになる可能性があります
    
    # ROIの中央値以上かどうか
    roi_median = df['roi'].median()
    df['is_high_roi'] = (df['roi'] > roi_median).astype(int)

    # --- ジャンルの処理 ---
    df['genre_list'] = df['genres'].apply(extract_genre_names)
    
    mlb = MultiLabelBinarizer()
    genre_dummies = mlb.fit_transform(df['genre_list'])
    
    # ジャンル名の整形 (スペースをアンダースコアに、Prefixを追加)
    genre_columns = [f"Genre_{cls.replace(' ', '_')}" for cls in mlb.classes_]
    genre_df = pd.DataFrame(genre_dummies, columns=genre_columns, index=df.index)

    # 上位10ジャンルのみ抽出して結合
    top_genres = genre_df.sum().sort_values(ascending=False).head(10).index
    df = pd.concat([df, genre_df[top_genres]], axis=1)

    # --- その他の特徴量 ---
    # コレクション有無
    df["belongs_to_collection"] = df["belongs_to_collection"].notna().astype(int)
    
    # ニュース数が100件ちょうどかどうか
    df['over_100news'] = (df['news_count'] == 100).astype(int)

    # Political Ratio のビン分割 (4等分以下に分ける)
    df['political_ratio_level'] = pd.qcut(
        df['political_ratio'], 
        q=4, 
        labels=False, 
        duplicates='drop'
    )

    return df