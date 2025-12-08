import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from data_processing import load_and_preprocess_data
from visualization import plot_distributions, plot_financials, plot_qq_and_reg, plot_heatmap, save_regression_summary
from visualization import analyze_threshold_sensitivity

# --- 設定 ---
INPUT_FILE = "data/movies_analyzed_3m.csv"

def main():
    # データ読み込み
    print("Loading and processing data...")
    df = load_and_preprocess_data(INPUT_FILE)

    # 可視化して画像を保存
    print("\nGenerating visualizations...")
    plot_distributions(df)
    plot_financials(df)
    plot_qq_and_reg(df)
    plot_heatmap(df)
    # ここに散布図
    
    # 共通フォーミュラ
    base_formula = """
        political_ratio * over_100news + budget_log + belongs_to_collection + 
        Genre_Drama + Genre_Comedy + Genre_Thriller + Genre_Action + 
        Genre_Adventure + Genre_Romance + Genre_Crime + Genre_Science_Fiction + 
        Genre_Family + Genre_Horror
    """

    # --- Model 1: Revenue Log ---
    print('\n========== Model 1: OLS for Revenue Log ==========')
    formula_rev = f"revenue_log ~ {base_formula}"
    model_rev = ols(formula_rev, data=df).fit()
    print(model_rev.summary())
    
    # 画像保存
    save_regression_summary(model_rev, "summary_model1_revenue.png")


    # --- Model 2: ROI Log ---
    print('\n========== Model 2: OLS for ROI Log ==========')
    formula_roi = f"roi_log ~ {base_formula}"
    model_roi = ols(formula_roi, data=df).fit()
    print(model_roi.summary())
    
    # 画像保存
    save_regression_summary(model_roi, "summary_model2_roi.png")


    # --- Model 3: Logistic Regression ---
    print('\n========== Model 3: Logistic Regression ==========')
    formula_logit = f"is_high_roi ~  {base_formula}"
    model_logit = logit(formula_logit, data=df).fit()
    print(model_logit.summary())
    
    # 画像保存
    save_regression_summary(model_logit, "summary_model3_logit.png")

    print('\n========== Sensitivity Analysis ==========')
    
    # 注目する変数（交互作用項）を指定
    target_variable = 'political_ratio:over_100news' 
    
    # 数式テンプレート (目的変数は関数内で書き換えるので dummy でOK)
    formula_template = f"dummy ~ {base_formula}"
    
    analyze_threshold_sensitivity(df, formula_template, target_variable)


if __name__ == "__main__":
    main()