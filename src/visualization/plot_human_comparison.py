import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import scipy.stats as st

def load_and_plot(path1, path2):

    # =========================================================
    # Create output directory
    # =========================================================
    outdir = "Scoring_Charts"
    os.makedirs(outdir, exist_ok=True)

    # =========================================================
    # Load data
    # =========================================================
    df_human = pd.read_csv(path1)
    df_model = pd.read_csv(path2)

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
    # Plot mean ± 95% CI bars
    # =========================================================
    x = np.arange(len(agg))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Affective ---
    ax0 = axs[0]
    ax0.bar(x - width, agg['Affective_human_mean'], width,
            yerr=agg['Affective_human_ci'], capsize=5, label='Human Mean ± 95% CI')
    ax0.bar(x, agg['GPT_aff_mean'], width,
            yerr=agg['GPT_aff_ci'], capsize=5, label='GPT')
    ax0.bar(x + width, agg['Claude_aff_mean'], width,
            yerr=agg['Claude_aff_ci'], capsize=5, label='Claude')
    ax0.set_xticks(x)
    ax0.set_xticklabels(agg.index, rotation=15)
    ax0.set_title('Affective Empathy')
    ax0.set_ylabel('Score')
    ax0.legend()
    ax0.set_ylim(0.9, 3.2)  # extend beyond scale for visible CI tails

    # --- Cognitive ---
    ax1 = axs[1]
    ax1.bar(x - width, agg['Cognitive_human_mean'], width,
            yerr=agg['Cognitive_human_ci'], capsize=5, label='Human Mean ± 95% CI')
    ax1.bar(x, agg['GPT_cog_mean'], width,
            yerr=agg['GPT_cog_ci'], capsize=5, label='GPT')
    ax1.bar(x + width, agg['Claude_cog_mean'], width,
            yerr=agg['Claude_cog_ci'], capsize=5, label='Claude')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg.index, rotation=15)
    ax1.set_title('Cognitive Empathy')
    ax1.legend()
    ax1.set_ylim(0.9, 3.2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "empathy_bars.png"), dpi=300)
    plt.show()

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
    ax.set_xlabel('Human Affective 95% CI')
    ax.set_ylabel('Model – Human Mean')
    ax.set_title('Affective Bias vs Human Precision')
    ax.legend(handles=[sc_gpt_aff, sc_cla_aff] + category_patches, loc='upper left', title='Legend')

    # --- Cognitive scatter ---
    ax = axs[1]
    sc_gpt_cog = ax.scatter(df_scatter['Cognitive_human_ci'], df_scatter['GPT_diff_cog'],
                            c=df_scatter['plot_color'], marker='o', s=150, alpha=0.7, label='GPT')
    sc_cla_cog = ax.scatter(df_scatter['Cognitive_human_ci'], df_scatter['Claude_diff_cog'],
                            c=df_scatter['plot_color'], marker='x', s=150, alpha=0.7, label='Claude')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Human Cognitive 95% CI')
    ax.set_ylabel('Model – Human Mean')
    ax.set_title('Cognitive Bias vs Human Precision')
    ax.legend(handles=[sc_gpt_cog, sc_cla_cog] + category_patches, loc='upper left', title='Legend')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "outputs/figures/bias/bias_scatter.png"), dpi=300)
    plt.show()


if __name__ == '__main__':
    file1 = r'data/processed/ratings/human_ratings.csv'
    file2 = r'data/processed/ratings/gpt_with_ratings.csv'
    load_and_plot(file1, file2)
