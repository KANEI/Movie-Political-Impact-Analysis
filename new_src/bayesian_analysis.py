import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import patsy
import matplotlib.pyplot as plt
from data_processing import load_and_preprocess_data
from visualization import save_arviz_plot, IMAGE_DIR

# --- 設定 ---
INPUT_FILE = "new_data/movies_with_news.csv"

def standardize_data(df, cols):
    """指定された列を標準化（Z-score normalization）する"""
    df_std = df.copy()
    for col in cols:
        if col in df_std.columns:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            df_std[f'{col}_std'] = (df_std[col] - mean_val) / std_val
    return df_std

def run_bayesian_model(df, formula, model_name, family='normal'):
    """
    PatsyとPyMCを用いてベイズモデルを構築・実行する共通関数
    family: 'normal' (線形回帰) or 'bernoulli' (ロジスティック回帰)
    """
    print(f"\n========== Running Bayesian Model: {model_name} ==========")
    
    # デザイン行列の作成
    y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
    X_columns = X.columns
    
    with pm.Model() as model:
        # 事前分布
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X.shape[1])
        
        # 線形予測子
        linear_pred = pm.math.dot(np.asarray(X), beta)
        
        # 尤度
        if family == 'normal':
            # 線形回帰の場合、誤差の分散 sigma も推定
            sigma = pm.HalfNormal("sigma", sigma=10)
            pm.Normal("y_obs", mu=linear_pred, sigma=sigma, observed=np.asarray(y).flatten())
            
        elif family == 'bernoulli':
            # ロジスティック回帰の場合、シグモイド変換
            p = pm.math.sigmoid(linear_pred)
            pm.Bernoulli("y_obs", p=p, observed=np.asarray(y).flatten())
        
        # MCMCサンプリング
        print("Sampling...")
        trace = pm.sample(draws=2000, tune=1000, chains=2, return_inferencedata=True)
        
    # 変数名をマッピング
    trace.posterior = trace.posterior.assign_coords({"beta_dim_0": X_columns})
    
    return trace, X_columns

def analyze_interaction(trace, interaction_name, model_name, family='normal'):
    """交互作用項の詳細分析と解釈"""
    print(f"\n--- Interaction Analysis: {interaction_name} ({model_name}) ---")
    
    try:
        beta_samples = trace.posterior["beta"].sel(beta_dim_0=interaction_name).values.flatten()
        
        # 確率計算
        prob_positive = (beta_samples > 0).mean()
        prob_negative = (beta_samples < 0).mean()
        
        hdi = az.hdi(beta_samples, hdi_prob=0.95)
        mean_val = beta_samples.mean()
        
        print(f"事後平均 (Mean): {mean_val:.4f}")
        print(f"95%信用区間 (HDI): {hdi}")
        
        if family == 'bernoulli':
            # ロジスティック回帰の場合はオッズ比も計算
            odds_samples = np.exp(beta_samples)
            prob_odds_gt_1 = (odds_samples > 1).mean()
            print(f"オッズ比 > 1 の確率: {prob_odds_gt_1:.1%}")
        else:
            print(f"係数が正である確率: {prob_positive:.1%}")
            print(f"係数が負である確率: {prob_negative:.1%}")

        # 簡易判定
        if prob_positive > 0.95:
            print(">> 結論: 強い「正の」影響が見られます。")
        elif prob_negative > 0.95:
            print(">> 結論: 強い「負の」影響が見られます。")
        else:
            print(">> 結論: 0を含んでおり、効果は不確実です。")
            
    except KeyError:
        print(f"Warning: Interaction term '{interaction_name}' not found in trace.")

def execute_bayesian_analysis():
    # データ読み込み
    print("Loading data...")
    df = load_and_preprocess_data(INPUT_FILE)
    
    # 標準化
    continuous_cols = ['political_news_count','actor_fame_log','budget_log']

    df_std = standardize_data(df, continuous_cols)
    
    # 確認表示
    print("\nStandardized columns summary:")
    print(df_std[[f'{c}_std' for c in continuous_cols if f'{c}_std' in df_std.columns]].describe().loc[['mean', 'std']])

    # 共通の右辺（説明変数）
    base_formula_rhs = """
        political_news_count + actor_fame_log + budget_log_std + belongs_to_collection + 
        Genre_Drama + Genre_Comedy + Genre_Thriller + 
        Genre_Action + Genre_Adventure + Genre_Romance + Genre_Crime + 
        Genre_Science_Fiction + Genre_Family + Genre_Horror
    """
    
    interaction_term = 'political_news_count'

    # ==========================================
    # Model 1: Revenue Log (Linear Regression)
    # ==========================================
    formula_rev = f"revenue_log ~ {base_formula_rhs}"
    trace_rev, cols_rev = run_bayesian_model(df_std, formula_rev, "Revenue Log", family='normal')
    
    # 要約表示
    print(az.summary(trace_rev, var_names=["beta"], kind="stats"))
    
    # 分析と保存
    analyze_interaction(trace_rev, interaction_term, "Revenue Log")
    save_arviz_plot(az.plot_trace, trace_rev, "bayes_trace_revenue.png", var_names=["beta", "sigma"])
    save_arviz_plot(az.plot_posterior, trace_rev, "bayes_post_revenue_interact.png", 
                    var_names=["beta"], coords={"beta_dim_0": interaction_term}, ref_val=0)

    # ==========================================
    # Model 2: ROI Log (Linear Regression)
    # ==========================================
    formula_roi = f"roi_log ~ {base_formula_rhs}"
    trace_roi, cols_roi = run_bayesian_model(df_std, formula_roi, "ROI Log", family='normal')
    
    analyze_interaction(trace_roi, interaction_term, "ROI Log")
    save_arviz_plot(az.plot_trace, trace_roi, "bayes_trace_roi.png", var_names=["beta", "sigma"])
    save_arviz_plot(az.plot_posterior, trace_roi, "bayes_post_roi_interact.png", 
                    var_names=["beta"], coords={"beta_dim_0": interaction_term}, ref_val=0)

    # ==========================================
    # Model 3: ROI dummy (Logistic Regression)
    # ==========================================
    formula_logit = f"is_high_roi ~ {base_formula_rhs}"
    trace_logit, cols_logit = run_bayesian_model(df_std, formula_logit, "High ROI (Logistic)", family='bernoulli')
    
    analyze_interaction(trace_logit, interaction_term, "High ROI (Logistic)", family='bernoulli')
    save_arviz_plot(az.plot_trace, trace_logit, "bayes_trace_logit.png", var_names=["beta"])
    save_arviz_plot(az.plot_posterior, trace_logit, "bayes_post_logit_interact.png", 
                    var_names=["beta"], coords={"beta_dim_0": interaction_term}, ref_val=0)

if __name__ == "__main__":
    execute_bayesian_analysis()