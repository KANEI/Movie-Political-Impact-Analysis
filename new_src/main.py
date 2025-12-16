import statsmodels.formula.api as smf
from data_processing import load_and_preprocess_data
from visualization import plot_distributions, plot_financials, plot_additional_exploratory_analysis, save_regression_summary
from visualization import analyze_threshold_sensitivity
from bayesian_analysis import execute_bayesian_analysis

INPUT_FILE = "new_data/movies_with_news.csv"

def main():
    print("Loading data...")
    df = load_and_preprocess_data(INPUT_FILE)
    
    # 可視化して画像を保存
    print("\nGenerating visualizations...")
    plot_distributions(df)
    plot_financials(df)
    plot_additional_exploratory_analysis(df)
    print(df[["actor_fame_log", "budget_log", "political_news_count"]].corr())

    # 共通フォーミュラ
    base_formula = """
        political_news_count + actor_fame_log + budget_log + belongs_to_collection + 
        Genre_Drama + Genre_Comedy + Genre_Thriller + 
        Genre_Action + Genre_Adventure + Genre_Romance + Genre_Crime + 
        Genre_Science_Fiction + Genre_Family + Genre_Horror
    """
    
    # Model 1: Revenue
    print("\n=== OLS: Revenue Log ===")
    f_rev = f"revenue_log ~ {base_formula}"
    model_rev = smf.ols(f_rev, data=df).fit()
    print(model_rev.summary())
    save_regression_summary(model_rev, "summary_new_method_revenue.png")

    # Model 2: ROI (Log)
    print("\n=== OLS: ROI Log ===")
    f_rev = f"roi_log ~ {base_formula}"
    model_rev = smf.ols(f_rev, data=df).fit()
    print(model_rev.summary())
    save_regression_summary(model_rev, "summary_new_method_roi.png")
    
    # Model 3: ROI (Logistic)
    print("\n=== Logit: High ROI ===")
    f_logit = f"is_high_roi ~ {base_formula}"
    model_logit = smf.logit(f_logit, data=df).fit()
    print(model_logit.summary())
    save_regression_summary(model_logit, "summary_new_method_logit.png")

    # Sensitivity Analysis
    print('\n========== Sensitivity Analysis ==========')
    target_variable = 'political_news_count' 
    formula_template = f"dummy ~ {base_formula}"
    analyze_threshold_sensitivity(df, formula_template, target_variable)

    # Bayesian Analysis
    execute_bayesian_analysis()

if __name__ == "__main__":
    main()