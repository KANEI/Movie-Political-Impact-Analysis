import pandas as pd
import ast

# 設定
INPUT_FILE = "data/movies_with_news_1m.csv"
OUTPUT_FILE = "data/movies_analyzed_1m.csv"

# 政治的な単語リスト
POLITICAL_KEYWORDS = [
    "politics", "political", "president", "election", "campaign", "vote",
    "democrat", "republican", "white house", "congress", "senate", "policy",
    "government", "activist", "protest", "scandal", "law", "rights",
    "obama", "trump", "bush", "clinton", "biden", "mccain", "romney"
]

def analyze_news_content(news_str):
    """
    ニュースリストを受け取り、以下の3つを返す

    """
    # CSVから読み込むと文字列になっているため、リスト型に変換
    try:
        titles = ast.literal_eval(news_str)
        if not isinstance(titles, list):
            return 0, 0, 0.0
    except (ValueError, SyntaxError):
        # 変換に失敗した場合は0を返す
        return 0, 0, 0.0

    total_count = len(titles)
    
    # ニュースが0件の場合はすべて0で返す
    if total_count == 0:
        return 0, 0, 0.0

    # 政治的な単語が含まれているかカウント
    political_count = 0
    for title in titles:
        # 大文字小文字を区別しないように小文字化してチェック
        title_lower = title.lower()
        
        # キーワードのいずれかがタイトルに含まれていればカウント
        if any(keyword in title_lower for keyword in POLITICAL_KEYWORDS):
            political_count += 1

    # 割合を計算
    ratio = political_count / total_count

    return total_count, political_count, ratio

def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print("Analyzing news titles...")

    # 全体的なニュースの数、政治的なニュースの数、その割合を算出
    analysis_results = df['news'].apply(lambda x: pd.Series(analyze_news_content(x)))
    
    # 列名を付ける
    analysis_results.columns = ['news_count', 'political_count', 'political_ratio']

    # 元のデータフレームと結合
    df_analyzed = pd.concat([df, analysis_results], axis=1)

    # news_count が 0 のものを除外する (news_count > 0 のデータのみ残す)
    initial_rows = len(df_analyzed)
    df_analyzed = df_analyzed[df_analyzed['news_count'] > 0]

    # 削除後の行数確認
    dropped_rows = initial_rows - len(df_analyzed)
    print(f"{dropped_rows}行のデータをフィルタリング。 残った行数：{len(df_analyzed)} ")

    # 結果の確認
    print("\n-----------")
    print(df_analyzed[['title', 'news_count', 'political_count', 'political_ratio']].head())
    print("\n-----------")

    # 保存
    print(f"\n次のCSVファイルとして保存：{OUTPUT_FILE}...")
    df_analyzed.to_csv(OUTPUT_FILE, index=False)
    print("完了！")

if __name__ == "__main__":
    main()