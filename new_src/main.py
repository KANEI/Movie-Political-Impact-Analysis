import pandas as pd
import statsmodels.formula.api as smf
from data_processing import load_and_preprocess_data
from visualization import (
    plot_distributions, plot_financials, 
    plot_additional_exploratory_analysis, save_regression_summary,
    analyze_threshold_sensitivity
)

INPUT_FILE = "new_data/movies_with_news.csv"

def main():
    print("Loading data...")
    df = load_and_preprocess_data(INPUT_FILE)
    
    # データの可視化
    print("\nGenerating visualizations...")
    plot_distributions(df)
    plot_financials(df)
    plot_additional_exploratory_analysis(df)

    # 比較したいモデル構成の定義
    # ジャンル変数の共通部分
    genres = "Genre_Drama + Genre_Comedy + Genre_Thriller + Genre_Action + Genre_Adventure + Genre_Romance + Genre_Crime + Genre_Science_Fiction + Genre_Family + Genre_Horror"
    
    formula_bases = {
        # 基本モデル
        "Base": f"political_news_count + actor_fame_log + budget_log + belongs_to_collection + {genres}",

        # 非線形性モデル
        "Quadratic": f"political_news_count + I(political_news_count**2) + actor_fame_log + budget_log + belongs_to_collection + {genres}",

        # 交差項
        "News_Fame": f"political_news_count * actor_fame_log + budget_log + belongs_to_collection + {genres}",
        "News_Budget": f"political_news_count * budget_log + actor_fame_log + belongs_to_collection + {genres}",
        "News_Collection": f"political_news_count * belongs_to_collection + actor_fame_log + budget_log + {genres}",

        # 3元交差項
        "News_Fame_Budget": f"political_news_count * actor_fame_log * budget_log + belongs_to_collection + {genres}",
        "News_Fame_Collection": f"political_news_count * actor_fame_log * belongs_to_collection + budget_log + {genres}",
        "News_Budget_Collection": f"political_news_count * budget_log * belongs_to_collection + actor_fame_log + {genres}",

        # フルモデル
        "Full_Complex": f"political_news_count * actor_fame_log * budget_log * belongs_to_collection + {genres}"
    }

    # 目的変数とモデルタイプの定義
    targets = [
        {"name": "Revenue", "dep_var": "revenue_log", "type": "ols"},
        {"name": "ROI", "dep_var": "roi_log", "type": "ols"},
        {"name": "High_ROI", "dep_var": "is_high_roi", "type": "logit"}
    ]

    # モデル比較の実行
    comparison_results = []

    for base_name, base_formula in formula_bases.items():
        print(f"\n{'='*20} Testing Base: {base_name} {'='*20}")
        
        for target in targets:
            full_formula = f"{target['dep_var']} ~ {base_formula}"
            print(f"\n--- Running {target['type'].upper()} for {target['name']} ---")
            
            try:
                if target['type'] == "ols":
                    model = smf.ols(full_formula, data=df).fit()

                else:
                    model = smf.logit(full_formula, data=df).fit()
                
                # 結果の表示と保存
                print(model.summary())
                file_name = f"summary_{base_name}_{target['name']}.png"
                save_regression_summary(model, file_name)
                
                # 比較用メトリクスの保存
                comparison_results.append({
                    "Base": base_name,
                    "Target": target['name'],
                    "AIC": model.aic,
                    "BIC": model.bic
                })
            
            except Exception as e:
                print(f"Error fitting {base_name} for {target['name']}: {e}")

    # 比較結果の要約表示
    print("\n" + "="*30)
    print("MODEL COMPARISON SUMMARY")
    print("="*30)
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.sort_values(by=["Target", "AIC"])) # AICが低い順に並び替え

    # 感度分析（代表として1つのフォーミュラで実行）
    print('\n========== Sensitivity Analysis ==========')
    sensitivity_formula = f"dummy ~ {formula_bases['News_Budget']}"
    analyze_threshold_sensitivity(df, sensitivity_formula, 'political_news_count:budget_log')

if __name__ == "__main__":
    main()