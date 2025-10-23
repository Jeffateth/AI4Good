# ==== Non-interactive + imports ====
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI popups
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- I/O paths ----
CLAUDE_CSV = "/Users/jianzhouyao/AI4Good/Ratings/Understandability/claude_responses_with_readability.csv"
GPT_CSV    = "/Users/jianzhouyao/AI4Good/Ratings/Understandability/gpt_responses_with_readability.csv"
OUTPUT_DIR = "/Users/jianzhouyao/AI4Good/Scoring_Charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pretty labels ↔ actual column names in your CSVs
metric_labels = ['Flesch-Kincaid\nGrade', 'SMOG\nIndex', 'Gunning Fog\nIndex',
                 'Coleman-Liau\nIndex', 'Dale-Chall\nScore']
metric_cols   = ['fk_grade', 'smog_index', 'gunning_fog_index',
                 'coleman_liau_index', 'dale_chall_score']

# Abbreviation mapping for conditions
COND_ABBR = {
    'pancreatic cancer': 'PanCan',
    'Pancreatic cancer': 'PanCan',
    'pancreatic Cancer': 'PanCan',
    'obesity': 'Obes',
    'Obesity': 'Obes',
    'Chronic Ischemic Heart Disease': 'CIHD',
    'chronic ischemic heart disease': 'CIHD',
    'Alzheimer\'s': 'Alz',
    'alzheimers': 'Alz',
}
DEFAULT_ABBR = 'Other'

# ---------- Load & prepare ----------
def find_col(possible, cols):
    for name in possible:
        if name in cols:
            return name
    return None

df_gpt = pd.read_csv(GPT_CSV);       df_gpt["model"] = "GPT"
df_claude = pd.read_csv(CLAUDE_CSV); df_claude["model"] = "Claude"
df = pd.concat([df_gpt, df_claude], ignore_index=True)

# Detect columns
age_col  = find_col(['age', 'Age'], df.columns)
cond_col = find_col(['medical_condition', 'condition', 'diagnosis',
                     'Medical Condition', 'Condition', 'Diagnosis'], df.columns)
id_col   = find_col(['Prompt Number','prompt_number','prompt_id','Prompt_ID'], df.columns)

# Normalize condition abbreviations
if cond_col:
    df['cond_abbr'] = df[cond_col].astype(str).map(COND_ABBR).fillna(DEFAULT_ABBR)

# Coerce metric cols to numeric
for c in metric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------- Bootstrap CI Helper ----------
def bootstrap_ci(data, n_bootstrap=1000, ci=95, random_seed=42):
    """
    Calculate bootstrap confidence interval for the mean.
    
    Parameters:
    - data: array-like of numeric values
    - n_bootstrap: number of bootstrap samples
    - ci: confidence level (e.g., 95 for 95% CI)
    - random_seed: for reproducibility
    
    Returns:
    - (lower_bound, upper_bound) as the CI half-width from the mean
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # remove NaN
    
    if len(data) < 2:
        return np.nan, np.nan
    
    np.random.seed(random_seed)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = (100 - ci) / 2
    lower_percentile = alpha
    upper_percentile = 100 - alpha
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    mean_val = np.mean(data)
    # Return as half-widths (distance from mean)
    return mean_val - ci_lower, ci_upper - mean_val

# ---------- Helpers ----------
def means_cis_by_category(dfin, group_series, categories_order):
    """Calculate means and 95% bootstrap CI half-widths by category"""
    cat_type = pd.CategoricalDtype(categories=categories_order, ordered=True)
    gser = group_series.astype('category').astype(cat_type)
    means, ci_lowers, ci_uppers = [], [], []
    
    for mc in metric_cols:
        grouped = dfin.groupby(gser, observed=False)[mc]
        
        m_list = []
        ci_lower_list = []
        ci_upper_list = []
        
        for cat in categories_order:
            if cat in grouped.groups:
                data = grouped.get_group(cat).dropna()
                if len(data) >= 2:
                    mean_val = data.mean()
                    ci_lower, ci_upper = bootstrap_ci(data.values)
                    m_list.append(mean_val)
                    ci_lower_list.append(ci_lower)
                    ci_upper_list.append(ci_upper)
                else:
                    m_list.append(np.nan)
                    ci_lower_list.append(np.nan)
                    ci_upper_list.append(np.nan)
            else:
                m_list.append(np.nan)
                ci_lower_list.append(np.nan)
                ci_upper_list.append(np.nan)
        
        means.append(m_list)
        ci_lowers.append(ci_lower_list)
        ci_uppers.append(ci_upper_list)
    
    # Convert to symmetric error bars (use the average of lower and upper)
    cis = []
    for lower_list, upper_list in zip(ci_lowers, ci_uppers):
        ci_symmetric = [(l + u) / 2 for l, u in zip(lower_list, upper_list)]
        cis.append(ci_symmetric)
    
    return means, cis

def smax(*mats):
    arrs = [np.array(m, dtype=float) for m in mats]
    return np.nanmax([np.nanmax(a) for a in arrs])

def finalize_and_save(fig, ax, outname, handles, labels, ncol):
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if handles and labels:
        ax.legend(handles, labels, loc="upper center",
                  bbox_to_anchor=(0.5, 1.18), fontsize=14,
                  framealpha=0.9, ncol=ncol)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    outpath = os.path.join(OUTPUT_DIR, outname)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# Simple Benjamini–Hochberg (FDR) for a list of p-values
def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    # largest k where p_k <= (k/n)*alpha
    if np.any(passed):
        kmax = np.max(np.where(passed)[0]) + 1
        cutoff = ranked[kmax-1]
        reject = pvals <= cutoff
    else:
        reject = np.zeros_like(pvals, dtype=bool)
    # adjusted p-values
    adj = np.empty_like(ranked)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        cummin = min(cummin, ranked[i] * n / (i+1))
        adj[i] = cummin
    p_adj = np.empty_like(pvals)
    p_adj[order] = np.minimum(adj, 1.0)
    return reject, p_adj

# Colormaps (keep your look)
cmap_gpt    = LinearSegmentedColormap.from_list("gpt_cmap", ["#6baed6", "#08519c"])
cmap_claude = LinearSegmentedColormap.from_list("claude_cmap", ["#fd8d3c", "#a63603"])

# ======================================================
# PLOT A: AGE GROUPS  (<18, 18–49, 50–64, 65+)
# ======================================================
if age_col:
    bins = [-float('inf'), 17, 49, 64, float('inf')]
    age_labels = ['<18', '18–49', '50–64', '65+']
    df['AgeGroup'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'),
                            bins=bins, labels=age_labels, right=True)

    df_gpt_only    = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for age groups...")
    gpt_means, gpt_cis       = means_cis_by_category(df_gpt_only, df_gpt_only['AgeGroup'], age_labels)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only['AgeGroup'], age_labels)

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(metric_labels))

    # Keep total cluster width to 0.8
    cats = age_labels
    total_slots = len(cats) * 2  # 2 models
    slot_width = 0.8 / total_slots

    handles, leglabels = [], []
    for model_idx, (name, means, cis, cmap) in enumerate([
        ("GPT", gpt_means, gpt_cis, cmap_gpt),
        ("Claude", claude_means, claude_cis, cmap_claude)
    ]):
        for i, cat in enumerate(cats):
            slot = model_idx * len(cats) + i
            offset = (slot - (total_slots - 1)/2) * slot_width
            color = cmap(i / (len(cats)-1)) if len(cats) > 1 else cmap(0.5)
            bars = ax.bar(
                x + offset,
                [means[m][i] for m in range(len(metric_labels))],
                slot_width,
                yerr=[cis[m][i] for m in range(len(metric_labels))],
                capsize=3,
                color=color,
                label=f"{name} - {cat}"
            )
            handles.append(bars[0]); leglabels.append(f"{name} - {cat}")

    ax.set_ylabel("Score", fontsize=24)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=22)
    ax.tick_params(axis="both", labelsize=22)

    ymax = smax(gpt_means, claude_means) + smax(gpt_cis, claude_cis) + 3
    if np.isfinite(ymax): ax.set_ylim(0, ymax)

    finalize_and_save(fig, ax, "readability_by_agegroup_gradient.png",
                      handles, leglabels, ncol=4)

# ======================================================
# PLOT B: MEDICAL CONDITION (with abbreviations)
# ======================================================
if cond_col:
    cond_order = []
    # prefer these four (in this order) if present; then append any "Other"
    preferred = ['PanCan','CIHD','Obes','Alz']
    present = df['cond_abbr'].dropna().astype(str).unique().tolist()
    for p in preferred:
        if p in present:
            cond_order.append(p)
    cond_order += sorted([c for c in present if c not in cond_order])

    df_gpt_only    = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for conditions...")
    gpt_means, gpt_cis       = means_cis_by_category(df_gpt_only, df_gpt_only['cond_abbr'].astype(str), cond_order)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only['cond_abbr'].astype(str), cond_order)

    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(metric_labels))

    cats = cond_order
    total_slots = max(1, len(cats)) * 2
    slot_width = 0.8 / total_slots

    handles, leglabels = [], []
    for model_idx, (name, means, cis, cmap) in enumerate([
        ("GPT", gpt_means, gpt_cis, cmap_gpt),
        ("Claude", claude_means, claude_cis, cmap_claude)
    ]):
        for i, cat in enumerate(cats):
            slot = model_idx * len(cats) + i
            offset = (slot - (total_slots - 1)/2) * slot_width
            color = cmap(i / (len(cats)-1)) if len(cats) > 1 else cmap(0.5)
            bars = ax.bar(
                x + offset,
                [means[m][i] for m in range(len(metric_labels))],
                slot_width,
                yerr=[cis[m][i] for m in range(len(metric_labels))],
                capsize=3,
                color=color,
                label=f"{name} - {cat}"
            )
            handles.append(bars[0]); leglabels.append(f"{name} - {cat}")

    ax.set_ylabel("Score", fontsize=24)
    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=22)
    ax.tick_params(axis="both", labelsize=22)

    ymax = smax(gpt_means, claude_means) + smax(gpt_cis, claude_cis) + 3
    if np.isfinite(ymax): ax.set_ylim(0, ymax)

    ncols = 3 if len(cond_order) > 6 else min(4, len(cond_order))
    finalize_and_save(fig, ax, "readability_by_condition_gradient.png",
                      handles, leglabels, ncol=ncols)

# ======================================================
# STATS: Paired t-tests (GPT vs Claude), BH-FDR corrected
# - per CONDITION (using cond_abbr)
# - per AGE GROUP
# Paired on the shared prompt id (Prompt Number)
# ======================================================
def paired_tests(df_gpt, df_claude, group_col, group_levels, label_name):
    if id_col is None:
        raise RuntimeError("No prompt ID column found to pair GPT/Claude.")
    results = []
    for level in group_levels:
        g_grp = df_gpt[df_gpt[group_col] == level]
        c_grp = df_claude[df_claude[group_col] == level]
        # inner merge on prompt id to ensure pairing
        merged = pd.merge(
            g_grp[[id_col] + metric_cols],
            c_grp[[id_col] + metric_cols],
            on=id_col,
            suffixes=('_gpt','_claude')
        )
        n = len(merged)
        if n < 2:
            # not enough pairs for a t-test
            for mc, label in zip(metric_cols, metric_labels):
                results.append({
                    label_name: level, 'metric': label, 'n_pairs': n,
                    'mean_gpt': np.nan, 'mean_claude': np.nan, 'mean_diff': np.nan,
                    't_stat': np.nan, 'p_raw': np.nan
                })
            continue

        for mc, label in zip(metric_cols, metric_labels):
            x = merged[f"{mc}_gpt"].astype(float)
            y = merged[f"{mc}_claude"].astype(float)
            # drop pairs with NaN in either
            mask = ~(x.isna() | y.isna())
            x = x[mask]; y = y[mask]
            n_eff = len(x)
            if n_eff < 2:
                t_stat = np.nan; p_val = np.nan
                mg = x.mean() if len(x) else np.nan
                mc_ = y.mean() if len(y) else np.nan
                mdiff = (x - y).mean() if len(x) and len(y) else np.nan
            else:
                # paired t-test by hand (scipy-free)
                d = (x - y).to_numpy()
                mdiff = float(np.mean(d))
                sd = float(np.std(d, ddof=1))
                t_stat = mdiff / (sd / np.sqrt(n_eff)) if sd > 0 else np.inf
                # two-sided p-value from t-dist via survival function approx (normal fallback)
                from math import erf, sqrt
                # normal approx if we avoid scipy: p ≈ 2*(1 - Phi(|t|))
                z = abs(t_stat)
                p_val = 2 * (1 - 0.5*(1 + erf(z / sqrt(2))))
                mg = float(x.mean()); mc_ = float(y.mean())
            results.append({
                label_name: level, 'metric': label, 'n_pairs': int(n_eff),
                'mean_gpt': mg, 'mean_claude': mc_, 'mean_diff': mdiff,
                't_stat': t_stat, 'p_raw': p_val
            })
    res_df = pd.DataFrame(results)

    # BH-FDR within this family
    mask_valid = res_df['p_raw'].notna()
    reject, p_adj = fdr_bh(res_df.loc[mask_valid, 'p_raw'].values, alpha=0.05)
    res_df.loc[mask_valid, 'p_adj'] = p_adj
    res_df['significant_fdr_0.05'] = False
    res_df.loc[mask_valid, 'significant_fdr_0.05'] = reject

    return res_df

# Build per-condition tests
if cond_col:
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    cond_levels = [c for c in df['cond_abbr'].dropna().astype(str).unique().tolist()]
    cond_levels = [c for c in ['PanCan','CIHD','Obes','Alz'] if c in cond_levels] + \
                  [c for c in cond_levels if c not in ['PanCan','CIHD','Obes','Alz']]
    cond_stats = paired_tests(df_g, df_c, 'cond_abbr', cond_levels, 'condition')
    cond_stats_path = os.path.join(OUTPUT_DIR, "stats_readability_by_condition.csv")
    cond_stats.to_csv(cond_stats_path, index=False)
    print(f"Saved stats: {cond_stats_path}")

# Build per-age-group tests
if age_col:
    age_levels = ['<18', '18–49', '50–64', '65+']
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    # ensure AgeGroup exists
    if 'AgeGroup' not in df.columns:
        bins = [-float('inf'), 17, 49, 64, float('inf')]
        df['AgeGroup'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'),
                                bins=bins, labels=age_levels, right=True)
    age_stats = paired_tests(df_g, df_c, 'AgeGroup', age_levels, 'age_group')
    age_stats_path = os.path.join(OUTPUT_DIR, "stats_readability_by_agegroup.csv")
    age_stats.to_csv(age_stats_path, index=False)
    print(f"Saved stats: {age_stats_path}")

# ---- Console summary (concise) ----
def brief_summary(df_stats, group_col):
    if df_stats is None or df_stats.empty:
        return
    alpha_note = "FDR q<0.05"
    print("\n=== Significant differences (GPT vs Claude) [{}] ===".format(alpha_note))
    for (grp, metric), sub in df_stats.groupby([group_col, 'metric']):
        row = sub.iloc[0]
        if bool(row.get('significant_fdr_0.05', False)):
            direction = "GPT>Claude" if row['mean_diff'] > 0 else "GPT<Claude"
            print(f"{group_col}={grp:>6s} | {metric:>22s} | n={int(row['n_pairs'])} | "
                  f"Δmean={row['mean_diff']:.2f} | t={row['t_stat']:.2f} | "
                  f"p={row['p_raw']:.3g} | q={row['p_adj']:.3g} | {direction}")

if cond_col:
    brief_summary(cond_stats, 'condition')
if age_col:
    brief_summary(age_stats, 'age_group')

print(f"\nAll charts + stats saved to: {OUTPUT_DIR}")