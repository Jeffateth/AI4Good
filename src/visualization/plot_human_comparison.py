import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import scipy.stats as st

def load_and_plot(path1, path2, path_ttest):

    # =========================================================
    # Create output directory
    # =========================================================
    outdir = "/Users/jianzhouyao/AI4Good/outputs/figures/human_LLM_comparison"
    os.makedirs(outdir, exist_ok=True)

    # =========================================================
    # Load data
    # =========================================================
    df_human = pd.read_csv(path1)
    df_model = pd.read_csv(path2)
    df_ttest = pd.read_csv(path_ttest)

    # ---------------------------------------------------------
    # Identify columns
    # ---------------------------------------------------------
    aff_cols = [c for c in df_human.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_human.columns if c.startswith('Cognitive_Empathy_Human_')]

    # =========================================================
    # Compute mean ± 95% CI per row (safe for empty rows)
    # =========================================================
    def mean_ci(series):
        arr = series.dropna().astype(float).values
        if len(arr) == 0:
            return np.nan, np.nan
        if len(arr) == 1:
            return arr[0], 0
        mean = np.mean(arr)
        sem = st.sem(arr)
        h = sem * st.t.ppf(0.975, len(arr) - 1)
        return mean, h

    df_human['Affective_human_mean'], df_human['Affective_human_ci'] = zip(
        *df_human[aff_cols].apply(mean_ci, axis=1)
    )
    df_human['Cognitive_human_mean'], df_human['Cognitive_human_ci'] = zip(
        *df_human[cog_cols].apply(mean_ci, axis=1)
    )

    # =========================================================
    # Merge with model dataframe
    # =========================================================
    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender']
    df = pd.merge(
        df_human[['Prompt Number', 'age', 'ethnicity', 'gender',
                  'Affective_human_mean', 'Affective_human_ci',
                  'Cognitive_human_mean', 'Cognitive_human_ci']],
        df_model,
        on=merge_cols
    )

    # =========================================================
    # Filter & prepare groups
    # =========================================================
    df = df[df['education'].str.lower() == 'high school diploma or lower']
    df['Category'] = df['ethnicity'] + ' ' + df['gender']
    categories = ['European female', 'African female', 'European male', 'African male']

    # =========================================================
    # Compute per-group means and 95% CIs
    # =========================================================
    def group_stats(series):
        arr = series.dropna().astype(float).values
        if len(arr) == 0:
            return np.nan, np.nan
        mean = np.mean(arr)
        if len(arr) == 1:
            return mean, 0
        sem = st.sem(arr)
        h = sem * st.t.ppf(0.975, len(arr) - 1)
        return mean, h

    agg = []
    for cat in categories:
        sub = df[df['Category'] == cat]
        if sub.empty:
            continue

        mean_aff_h, ci_aff_h = group_stats(sub['Affective_human_mean'])
        mean_aff_g, ci_aff_g = group_stats(sub['Affective Empathy Score (GPT)'])
        mean_aff_c, ci_aff_c = group_stats(sub['Affective Empathy Score (Claude)'])
        mean_cog_h, ci_cog_h = group_stats(sub['Cognitive_human_mean'])
        mean_cog_g, ci_cog_g = group_stats(sub['Cognitive Empathy Score (GPT)'])
        mean_cog_c, ci_cog_c = group_stats(sub['Cognitive Empathy Score (Claude)'])

        agg.append({
            'Category': cat,
            'Affective_human_mean': mean_aff_h, 'Affective_human_ci': ci_aff_h,
            'GPT_aff_mean': mean_aff_g, 'GPT_aff_ci': ci_aff_g,
            'Claude_aff_mean': mean_aff_c, 'Claude_aff_ci': ci_aff_c,
            'Cognitive_human_mean': mean_cog_h, 'Cognitive_human_ci': ci_cog_h,
            'GPT_cog_mean': mean_cog_g, 'GPT_cog_ci': ci_cog_g,
            'Claude_cog_mean': mean_cog_c, 'Claude_cog_ci': ci_cog_c,
        })

    agg = pd.DataFrame(agg).set_index('Category').reindex(categories)

    # =========================================================
    # Extract p-values from t-test results
    # =========================================================
    def get_significance_marker(p_value):
        """Convert p-value to significance marker"""
        if pd.isna(p_value):
            return ''
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'
    
    # Create dictionaries for quick lookup of p-values
    # For within-category comparisons
    sig_dict = {}
    for _, row in df_ttest[df_ttest['TestType'] == 'Paired (within category)'].iterrows():
        cat = row['Category']
        comp = row['Comparison']
        p_val = row['p_value']
        sig_dict[(cat, comp)] = get_significance_marker(p_val)

    # For between-group comparisons (African female vs European female)
    between_group_sig = {}
    for _, row in df_ttest[df_ttest['TestType'] == 'Two-sample (between groups)'].iterrows():
        comp = row['Comparison']
        p_val = row['p_value']
        between_group_sig[comp] = get_significance_marker(p_val)

    # =========================================================
    # Plot mean ± 95% CI bars WITH SIGNIFICANCE MARKERS
    # =========================================================
    x = np.arange(len(agg))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Affective ---
    ax0 = axs[0]
    bars_h_aff = ax0.bar(x - width, agg['Affective_human_mean'], width,
                         yerr=agg['Affective_human_ci'], capsize=5, 
                         label='Human Mean ± 95% CI', color='C0')
    bars_g_aff = ax0.bar(x, agg['GPT_aff_mean'], width,
                         yerr=agg['GPT_aff_ci'], capsize=5, 
                         label='GPT', color='C1')
    bars_c_aff = ax0.bar(x + width, agg['Claude_aff_mean'], width,
                         yerr=agg['Claude_aff_ci'], capsize=5, 
                         label='Claude', color='C2')
    
    # Add significance markers with brackets for Affective (within category)
    max_bracket_height = 0  # Track the highest bracket to position between-group bracket above
    for i, cat in enumerate(agg.index):
        # GPT vs Human
        sig_gpt = sig_dict.get((cat, 'Affective human vs GPT'), '')
        if sig_gpt and sig_gpt != 'ns':
            # Get the heights for the two bars being compared
            h1 = agg.loc[cat, 'Affective_human_mean'] + agg.loc[cat, 'Affective_human_ci']
            h2 = agg.loc[cat, 'GPT_aff_mean'] + agg.loc[cat, 'GPT_aff_ci']
            y_max = max(h1, h2)
            
            # Draw bracket
            x1, x2 = x[i] - width, x[i]  # Human bar to GPT bar
            bracket_y = y_max + 0.08
            max_bracket_height = max(max_bracket_height, bracket_y + 0.02)
            ax0.plot([x1, x1, x2, x2], [y_max + 0.03, bracket_y, bracket_y, y_max + 0.03], 
                    'k-', linewidth=1.5)
            
            # Add significance marker above bracket
            ax0.text((x1 + x2) / 2, bracket_y + 0.02, sig_gpt, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        
        # Claude vs Human
        sig_claude = sig_dict.get((cat, 'Affective human vs Claude'), '')
        if sig_claude and sig_claude != 'ns':
            # Get the heights for the two bars being compared
            h1 = agg.loc[cat, 'Affective_human_mean'] + agg.loc[cat, 'Affective_human_ci']
            h3 = agg.loc[cat, 'Claude_aff_mean'] + agg.loc[cat, 'Claude_aff_ci']
            y_max = max(h1, h3)
            
            # Draw bracket
            x1, x2 = x[i] - width, x[i] + width  # Human bar to Claude bar
            bracket_y = y_max + 0.08
            max_bracket_height = max(max_bracket_height, bracket_y + 0.02)
            ax0.plot([x1, x1, x2, x2], [y_max + 0.03, bracket_y, bracket_y, y_max + 0.03], 
                    'k-', linewidth=1.5)
            
            # Add significance marker above bracket
            ax0.text((x1 + x2) / 2, bracket_y + 0.02, sig_claude, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
    
    # Add between-group significance markers (European female vs African female) - Affective
    euro_idx = list(agg.index).index('European female')
    afr_idx = list(agg.index).index('African female')
    
    # For Human Affective
    sig_human_aff = between_group_sig.get('Affective human', '')
    if sig_human_aff and sig_human_aff != 'ns':
        # Position bracket above all within-category brackets
        bracket_y = max_bracket_height + 0.15
        
        # Draw horizontal line connecting the two groups
        x1, x2 = x[euro_idx] - width, x[afr_idx] - width
        ax0.plot([x1, x2], [bracket_y, bracket_y], 'k-', linewidth=1.5)
        
        # Draw vertical lines down from bracket
        h_euro = agg.loc['European female', 'Affective_human_mean'] + agg.loc['European female', 'Affective_human_ci']
        h_afr = agg.loc['African female', 'Affective_human_mean'] + agg.loc['African female', 'Affective_human_ci']
        ax0.plot([x1, x1], [max(h_euro, max_bracket_height - 0.05), bracket_y], 'k-', linewidth=1.5)
        ax0.plot([x2, x2], [max(h_afr, max_bracket_height - 0.05), bracket_y], 'k-', linewidth=1.5)
        
        # Add significance marker
        ax0.text((x1 + x2) / 2, bracket_y + 0.02, sig_human_aff, ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax0.set_xticks(x)
    ax0.set_xticklabels(agg.index, rotation=15, ha='right', fontsize=11)
    ax0.set_title('Affective Empathy', fontsize=16, fontweight='bold', pad=15)
    ax0.set_ylabel('Score', fontsize=13)
    ax0.set_ylim(0.9, 3.8)  # Extended to accommodate all significance markers

    # --- Cognitive ---
    ax1 = axs[1]
    bars_h_cog = ax1.bar(x - width, agg['Cognitive_human_mean'], width,
                         yerr=agg['Cognitive_human_ci'], capsize=5, 
                         label='Human Mean ± 95% CI', color='C0')
    bars_g_cog = ax1.bar(x, agg['GPT_cog_mean'], width,
                         yerr=agg['GPT_cog_ci'], capsize=5, 
                         label='GPT', color='C1')
    bars_c_cog = ax1.bar(x + width, agg['Claude_cog_mean'], width,
                         yerr=agg['Claude_cog_ci'], capsize=5, 
                         label='Claude', color='C2')
    
    # Add significance markers with brackets for Cognitive (within category)
    for i, cat in enumerate(agg.index):
        # GPT vs Human
        sig_gpt = sig_dict.get((cat, 'Cognitive human vs GPT'), '')
        if sig_gpt and sig_gpt != 'ns':
            # Get the heights for the two bars being compared
            h1 = agg.loc[cat, 'Cognitive_human_mean'] + agg.loc[cat, 'Cognitive_human_ci']
            h2 = agg.loc[cat, 'GPT_cog_mean'] + agg.loc[cat, 'GPT_cog_ci']
            y_max = max(h1, h2)
            
            # Draw bracket
            x1, x2 = x[i] - width, x[i]  # Human bar to GPT bar
            bracket_y = y_max + 0.08
            ax1.plot([x1, x1, x2, x2], [y_max + 0.03, bracket_y, bracket_y, y_max + 0.03], 
                    'k-', linewidth=1.5)
            
            # Add significance marker above bracket
            ax1.text((x1 + x2) / 2, bracket_y + 0.02, sig_gpt, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        
        # Claude vs Human
        sig_claude = sig_dict.get((cat, 'Cognitive human vs Claude'), '')
        if sig_claude and sig_claude != 'ns':
            # Get the heights for the two bars being compared
            h1 = agg.loc[cat, 'Cognitive_human_mean'] + agg.loc[cat, 'Cognitive_human_ci']
            h3 = agg.loc[cat, 'Claude_cog_mean'] + agg.loc[cat, 'Claude_cog_ci']
            y_max = max(h1, h3)
            
            # Draw bracket
            x1, x2 = x[i] - width, x[i] + width  # Human bar to Claude bar
            bracket_y = y_max + 0.08
            ax1.plot([x1, x1, x2, x2], [y_max + 0.03, bracket_y, bracket_y, y_max + 0.03], 
                    'k-', linewidth=1.5)
            
            # Add significance marker above bracket
            ax1.text((x1 + x2) / 2, bracket_y + 0.02, sig_claude, ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
    
    # No significant between-group differences for Cognitive empathy
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg.index, rotation=15, ha='right', fontsize=11)
    ax1.set_title('Cognitive Empathy', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim(0.9, 3.8)

    # Add single legend at the bottom center for both panels
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 0.10), frameon=True)

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # give more bottom space
    fig.text(0.5, 0.03, '* p < 0.05    ** p < 0.01    *** p < 0.001',
            ha='center', fontsize=11, style='italic')

    plt.savefig(os.path.join(outdir, "empathy_bars_with_significance.png"), dpi=300)
    plt.show()

    # =========================================================
    # Print significance summary
    # =========================================================
    print("\n" + "="*70)
    print("SIGNIFICANCE SUMMARY - WITHIN CATEGORY COMPARISONS")
    print("="*70)
    for cat in categories:
        print(f"\n{cat}:")
        print(f"  Affective - GPT vs Human: {sig_dict.get((cat, 'Affective human vs GPT'), 'N/A')}")
        print(f"  Affective - Claude vs Human: {sig_dict.get((cat, 'Affective human vs Claude'), 'N/A')}")
        print(f"  Cognitive - GPT vs Human: {sig_dict.get((cat, 'Cognitive human vs GPT'), 'N/A')}")
        print(f"  Cognitive - Claude vs Human: {sig_dict.get((cat, 'Cognitive human vs Claude'), 'N/A')}")
    
    print("\n" + "="*70)
    print("SIGNIFICANCE SUMMARY - BETWEEN GROUP COMPARISONS")
    print("="*70)
    print("African female vs European female:")
    for comp, sig in between_group_sig.items():
        print(f"  {comp}: {sig}")
    print("="*70 + "\n")

    # =========================================================
    # Scatterplots: Bias vs Human CI
    # =========================================================
    df['GPT_diff_aff']    = df['Affective Empathy Score (GPT)']    - df['Affective_human_mean']
    df['Claude_diff_aff'] = df['Affective Empathy Score (Claude)'] - df['Affective_human_mean']
    df['GPT_diff_cog']    = df['Cognitive Empathy Score (GPT)']    - df['Cognitive_human_mean']
    df['Claude_diff_cog'] = df['Cognitive Empathy Score (Claude)'] - df['Cognitive_human_mean']

    df_scatter = df[df['Category'].isin(['European female', 'African female'])].copy()
    print("Scatter points:", len(df_scatter))

    color_map = {'European female': 'C0', 'African female': 'C1'}
    df_scatter['plot_color'] = df_scatter['Category'].map(color_map)

    category_patches = [
        mpatches.Patch(color='C0', label='European female'),
        mpatches.Patch(color='C1', label='African female')
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Affective scatter ---
    ax = axs[0]
    sc_gpt_aff = ax.scatter(df_scatter['Affective_human_ci'], df_scatter['GPT_diff_aff'],
                            c=df_scatter['plot_color'], marker='o', s=150, alpha=0.7, label='GPT')
    sc_cla_aff = ax.scatter(df_scatter['Affective_human_ci'], df_scatter['Claude_diff_aff'],
                            c=df_scatter['plot_color'], marker='x', s=150, alpha=0.7, label='Claude')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Human Affective 95% CI', fontsize=12)
    ax.set_ylabel('Model – Human Mean', fontsize=12)
    ax.set_title('Affective Bias vs Human Precision', fontsize=14, fontweight='bold')
    ax.legend(handles=[sc_gpt_aff, sc_cla_aff] + category_patches, loc='upper left', title='Legend')

    # --- Cognitive scatter ---
    ax = axs[1]
    sc_gpt_cog = ax.scatter(df_scatter['Cognitive_human_ci'], df_scatter['GPT_diff_cog'],
                            c=df_scatter['plot_color'], marker='o', s=150, alpha=0.7, label='GPT')
    sc_cla_cog = ax.scatter(df_scatter['Cognitive_human_ci'], df_scatter['Claude_diff_cog'],
                            c=df_scatter['plot_color'], marker='x', s=150, alpha=0.7, label='Claude')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Human Cognitive 95% CI', fontsize=12)
    ax.set_ylabel('Model – Human Mean', fontsize=12)
    ax.set_title('Cognitive Bias vs Human Precision', fontsize=14, fontweight='bold')
    ax.legend(handles=[sc_gpt_cog, sc_cla_cog] + category_patches, loc='upper left', title='Legend')

    plt.tight_layout()
    
    # Save to the same output directory
    plt.savefig(os.path.join(outdir, "bias_scatter.png"), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    file1 = r'/Users/jianzhouyao/AI4Good/data/processed/ratings/human_ratings.csv'
    file2 = r'/Users/jianzhouyao/AI4Good/data/processed/ratings/gpt_with_ratings.csv'
    file3 = r'/Users/jianzhouyao/AI4Good/data/results/statistical_tests/human_empathy_t_tests.csv'
    load_and_plot(file1, file2, file3)