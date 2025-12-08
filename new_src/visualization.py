import os
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 保存先フォルダの設定
IMAGE_DIR = "new_image"

# フォルダがなければ作成する
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def save_plot(filename):
    """グラフを保存し、表示するヘルパー関数"""
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    plt.show()
    plt.close() # メモリ解放のためクローズ

def save_regression_summary(model, filename):
    """
    回帰分析の結果(summary)を画像として保存する
    """
    # モデルの結果をテキストとして取得
    summary_text = model.summary().as_text()
    
    # 描画エリアを作成 (文字数に合わせてサイズを調整すると良いですが、ここでは大きめに確保)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 軸を消す
    ax.axis('off')
    
    # テキストを描画
    # fontfamily='monospace' にしないと、表の桁がズレます
    ax.text(0.01, 0.99, summary_text, 
            fontsize=10, 
            fontfamily='monospace', 
            verticalalignment='top', 
            horizontalalignment='left')
    
    save_path = os.path.join(IMAGE_DIR, filename)
    
    # 保存 (bbox_inches='tight' で白い余白をカット)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    print(f"Saved summary to: {save_path}")
    plt.close()

def plot_distributions(df):
    """
    ニュース数や政治的指標の分布を描画
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    
    columns = ['actor_fame', 'actor_fame_log', 'political_news_count']
    colors = ['skyblue', 'salmon', 'salmon']
    titles = ['Actor Fame', 'Actor Fame Log', 'Political News Count']

    for ax, col, color, title in zip(axes, columns, colors, titles):
        ax.hist(df[col], bins=20, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(f'{title} Histogram')
        ax.set_xlabel(title)
        ax.set_ylabel('Count')

    save_plot("distributions.png")

def plot_additional_exploratory_analysis(df):
    """
    追加の探索的データ分析（散布図、相関、箱ひげ図）を行い保存する
    """
    print("\n--- Generating Additional Exploratory Plots ---")

    # 箱ひげ図: political_news_count
    plt.figure(figsize=(6, 8))
    # boxplotはNaNが含まれていると描画できない場合があるためdropnaする
    plt.boxplot(df['political_news_count'].dropna(), patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    plt.title('Boxplot: Political News Count')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot("boxplot_political.png")

    # 箱ひげ図: actor_fame_log
    plt.figure(figsize=(6, 8))
    plt.boxplot(df['actor_fame_log'].dropna(), patch_artist=True,
                boxprops=dict(facecolor="lightgreen"))
    plt.title('Boxplot: Actor Fame Log')
    plt.ylabel('Log Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot("boxplot_fame.png")

    # 散布図: political_news_count vs revenue_log
    plt.figure(figsize=(8, 6))
    plt.scatter(df['political_news_count'], df['revenue_log'], alpha=0.6, edgecolors='w', s=60)
    plt.title('Scatter: Political News Count vs Revenue Log')
    plt.xlabel('Political News Count')
    plt.ylabel('Revenue Log')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot("scatter_political_revenue.png")

    # 散布図: political_news_count vs actor_fame_log
    plt.figure(figsize=(8, 6))
    plt.scatter(df['political_news_count'], df['actor_fame_log'], alpha=0.6, color='orange', edgecolors='w', s=60)
    plt.title('Scatter: Political News Count vs Actor Fame Log')
    plt.xlabel('Political News Count')
    plt.ylabel('Actor Fame Log')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot("scatter_political_fame.png")

    # 相関行列の計算と表示
    corr_cols = ["political_news_count", 'actor_fame_log']
    corr_mat = df[corr_cols].corr()
    print("\n[Correlation Matrix]")
    print(corr_mat)
    
    ## 相関行列をヒートマップとして保存
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ## 軸ラベルの設定
    ticks = np.arange(len(corr_cols))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_cols, rotation=15, ha='left')
    ax.set_yticklabels(corr_cols)
    
    ## 値をセル内に表示
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr_mat.iloc[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=12)
            
    plt.title("Correlation Matrix", pad=20)
    save_plot("correlation_heatmap.png")

    

def analyze_threshold_sensitivity(df, formula_template, target_term, quantiles=None):
    """
    閾値を変動させてロジスティック回帰を行い、係数の安定性をプロットする
    """
    if quantiles is None:
        quantiles = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    results = []
    
    # ベースの数式の右辺だけ取り出す
    rhs = formula_template.split('~')[1]
    
    print(f"--- Sensitivity Analysis for {target_term} ---")
    
    for q in quantiles:
        # 閾値を計算
        threshold_val = df['roi'].quantile(q)
        
        # 一時的な目的変数を作成 (閾値を超えたら1)
        temp_col_name = f'is_high_roi_q{int(q*100)}'
        df[temp_col_name] = (df['roi'] > threshold_val).astype(int)
        
        # モデル構築
        current_formula = f"{temp_col_name} ~ {rhs}"
        try:
            model = smf.logit(current_formula, data=df).fit(disp=0) # disp=0でログ抑制
            
            # 係数と標準誤差、P値を取得
            coef = model.params[target_term]
            se = model.bse[target_term]
            p_val = model.pvalues[target_term]
            
            # 95%信頼区間 (近似的に 1.96 * SE)
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            
            results.append({
                'quantile': q,
                'coef': coef,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_val
            })
            print(f"Quantile {q:.2f}: Coef={coef:.4f}, P-val={p_val:.4f}")
            
        except Exception as e:
            print(f"Quantile {q:.2f}: Error - {e}")

    # 結果をDataFrame化
    res_df = pd.DataFrame(results)
    
    # --- 可視化 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 係数のプロット（信頼区間付き）
    ax1.errorbar(
        res_df['quantile'], 
        res_df['coef'], 
        yerr=[res_df['coef'] - res_df['ci_lower'], res_df['ci_upper'] - res_df['coef']], 
        fmt='-o', 
        capsize=5, 
        color='blue',
        label='Coefficient (Effect Size)'
    )
    
    # ゼロライン（ここを跨ぐと有意ではない）
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    ax1.set_xlabel('ROI Threshold Quantile (Median=0.5)')
    ax1.set_ylabel('Coefficient Estimate')
    ax1.set_title(f'Sensitivity Analysis: {target_term}')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # P値を右軸に表示（オプション）
    ax2 = ax1.twinx()
    ax2.plot(res_df['quantile'], res_df['p_value'], color='red', linestyle=':', marker='x', label='P-value')
    ax2.set_ylabel('P-value')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, linewidth=1) # 有意水準5%線
    
    # 凡例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 保存
    from visualization import IMAGE_DIR, os # インポートのスコープに注意
    save_path = os.path.join(IMAGE_DIR, "sensitivity_analysis.png")
    plt.savefig(save_path)
    print(f"Saved sensitivity plot to: {save_path}")
    plt.show()

def save_arviz_plot(plot_func, trace, filename, **kwargs):
    """
    ArviZのプロット関数を実行して保存するラッパー関数
    """
    # プロットの実行
    ax = plot_func(trace, **kwargs)
    
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved ArviZ plot to: {save_path}")
    plt.close() # メモリ解放