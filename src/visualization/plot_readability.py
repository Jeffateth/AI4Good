# ==== Non-interactive + imports ====
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI popups
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------ Style constants (match overall plot) ------------------
YLABEL_FONTSIZE = 32
XTICK_FONTSIZE  = 28
TICK_FONTSIZE   = 28
LEGEND_FONTSIZE = 24
VAL_LABEL_FSIZE = 22
ERR_CAPSIZE     = 5

# ---- I/O paths ----
# Use your repo paths by default; if not found, fall back to /mnt/data (for local testing).
def pick_path(primary, fallback):
    return primary if os.path.exists(primary) else fallback

CLAUDE_CSV = pick_path("data/processed/readability/claude_readability.csv",
                       "/mnt/data/claude_readability.csv")
GPT_CSV    = pick_path("data/processed/readability/gpt_readability.csv",
                       "/mnt/data/gpt_readability.csv")

OUTPUT_DIR = "."  # Script runs from repo root
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pretty labels ↔ actual column names in your CSVs
metric_labels = ['Flesch-Kincaid\nGrade', 'SMOG\nIndex', 'Gunning Fog\nIndex',
                 'Coleman-Liau\nIndex', 'Dale-Chall\nScore']
metric_cols   = ['fk_grade', 'smog_index', 'gunning_fog_index',
                 'coleman_liau_index', 'dale_chall_score']

# Abbreviation mapping for medical conditions (unchanged)
COND_ABBR = {
    'pancreatic cancer': 'PanCan',
    'Pancreatic cancer': 'PanCan',
    'pancreatic Cancer': 'PanCan',
    'obesity': 'Obes',
    'Obesity': 'Obes',
    'Chronic Ischemic Heart Disease': 'CIHD',
    'chronic ischemic heart disease': 'CIHD',
    "Alzheimer’s": 'Alz',
}
DEFAULT_ABBR = 'Other'

# ------------------ Load & prepare ------------------
def find_col(possible, cols):
    for name in possible:
        if name in cols:
            return name
    return None

df_gpt = pd.read_csv(GPT_CSV);       df_gpt["model"] = "GPT"
df_claude = pd.read_csv(CLAUDE_CSV); df_claude["model"] = "Claude"
df = pd.concat([df_gpt, df_claude], ignore_index=True)

# Detect columns (present in your CSVs)
age_col  = find_col(['age'], df.columns)
gender_col = find_col(['gender'], df.columns)
education_col = find_col(['education'], df.columns)
race_col = find_col(['ethnicity'], df.columns)
cond_col = find_col(['diagnosis'], df.columns)
id_col   = find_col(['Prompt Number'], df.columns)

# Normalize condition abbreviations
if cond_col:
    df['cond_abbr'] = df[cond_col].astype(str).map(COND_ABBR).fillna(DEFAULT_ABBR)

# Coerce metric cols to numeric
for c in metric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------- Bootstrap CI Helper ----------
def bootstrap_ci(data, n_bootstrap=1000, ci=95, random_seed=42):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if data.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    means = np.empty(n_bootstrap, dtype=float)
    n = data.size
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = sample.mean()
    alpha = (100 - ci) / 2.0
    lower = np.percentile(means, alpha)
    upper = np.percentile(means, 100 - alpha)
    mean_val = data.mean()
    return mean_val - lower, upper - mean_val  # asymmetric distances

# ---------- Helpers ----------
def means_cis_by_category(dfin, group_series, categories_order):
    # keep order; any category not present becomes NaN row
    means, ci_lowers, ci_uppers = [], [], []
    grouped = dfin.groupby(group_series.astype(str), observed=False)

    for mc in metric_cols:
        m_list, lo_list, up_list = [], [], []
        for cat in categories_order:
            if cat in grouped.groups:
                data = pd.to_numeric(grouped.get_group(cat)[mc], errors='coerce').dropna().values
                if data.size >= 2:
                    mean_val = float(np.mean(data))
                    ci_l, ci_u = bootstrap_ci(data)
                else:
                    mean_val = np.nan; ci_l = np.nan; ci_u = np.nan
            else:
                mean_val = np.nan; ci_l = np.nan; ci_u = np.nan
            m_list.append(mean_val); lo_list.append(ci_l); up_list.append(ci_u)
        means.append(m_list); ci_lowers.append(lo_list); ci_uppers.append(up_list)

    # Convert to symmetric CI half-width for matplotlib yerr
    cis = []
    for lower_list, upper_list in zip(ci_lowers, ci_uppers):
        cis.append([(l + u) / 2 if (np.isfinite(l) and np.isfinite(u)) else np.nan
                    for l, u in zip(lower_list, upper_list)])
    return means, cis

def smax(*mats):
    # safe nanmax across lists-of-lists
    vals = []
    for m in mats:
        a = np.array(m, dtype=float)
        if a.size:
            vals.append(np.nanmax(a))
    return np.nanmax(vals) if vals else np.nan

def _safe_ylim(*pairs, pad=3):
    m = smax(*[p[0] for p in pairs])
    c = smax(*[p[1] for p in pairs])
    if np.isfinite(m) and np.isfinite(c):
        return 0, m + c + pad
    elif np.isfinite(m):
        return 0, m + pad
    return None

def finalize_and_save(fig, ax, outname, handles, labels, ncol, legend_fontsize=LEGEND_FONTSIZE):
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    if handles and labels:
        ax.legend(handles, labels, loc="upper left",
                  fontsize=legend_fontsize, framealpha=0.0, ncol=ncol,
                  edgecolor='black', fancybox=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outname}")

# Colormaps
cmap_gpt    = LinearSegmentedColormap.from_list("gpt_cmap", ["#6baed6", "#08519c"])
cmap_claude = LinearSegmentedColormap.from_list("claude_cmap", ["#fd8d3c", "#a63603"])
gpt_color   = "#2171b5"
claude_color= "#d94801"

# ======================================================
# PLOT 0: OVERALL UNDERSTANDABILITY (with value labels)
# ======================================================
print("Calculating overall readability metrics...")

df_gpt_overall = df[df["model"] == "GPT"]
df_claude_overall = df[df["model"] == "Claude"]

gpt_overall_means, gpt_overall_cis = [], []
claude_overall_means, claude_overall_cis = [], []

for mc in metric_cols:
    gpt_data = df_gpt_overall[mc].dropna().values
    if gpt_data.size >= 2:
        gpt_overall_means.append(float(np.mean(gpt_data)))
        l,u = bootstrap_ci(gpt_data); gpt_overall_cis.append((l+u)/2)
    else:
        gpt_overall_means.append(np.nan); gpt_overall_cis.append(np.nan)

    claude_data = df_claude_overall[mc].dropna().values
    if claude_data.size >= 2:
        claude_overall_means.append(float(np.mean(claude_data)))
        l,u = bootstrap_ci(claude_data); claude_overall_cis.append((l+u)/2)
    else:
        claude_overall_means.append(np.nan); claude_overall_cis.append(np.nan)

fig, ax = plt.subplots(figsize=(16, 9))
x = np.arange(len(metric_labels))
width = 0.35

bars1 = ax.bar(x - width/2, gpt_overall_means, width,
               yerr=gpt_overall_cis, capsize=ERR_CAPSIZE,
               color=gpt_color, label='GPT', alpha=0.9)
bars2 = ax.bar(x + width/2, claude_overall_means, width,
               yerr=claude_overall_cis, capsize=ERR_CAPSIZE,
               color=claude_color, label='Claude', alpha=0.9)

# Value labels
for i, (gv, gc, cv, cc) in enumerate(zip(gpt_overall_means, gpt_overall_cis,
                                         claude_overall_means, claude_overall_cis)):
    if np.isfinite(gv) and np.isfinite(gc):
        ax.text(i - width/2, gv + gc + 0.5, f'{gv:.2f}', ha='center', va='bottom', fontsize=VAL_LABEL_FSIZE)
    if np.isfinite(cv) and np.isfinite(cc):
        ax.text(i + width/2, cv + cc + 0.5, f'{cv:.2f}', ha='center', va='bottom', fontsize=VAL_LABEL_FSIZE)

ax.set_ylabel("Score", fontsize=YLABEL_FONTSIZE)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=XTICK_FONTSIZE)
ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', framealpha=0.0, edgecolor='black')
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ylim = _safe_ylim((gpt_overall_means, gpt_overall_cis), (claude_overall_means, claude_overall_cis))
if ylim: ax.set_ylim(*ylim)

fig.tight_layout()
overall_path = "outputs/figures/readability/overall.png"
os.makedirs(os.path.dirname(overall_path), exist_ok=True)
fig.savefig(overall_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {overall_path}")

# ------------------ Generic demographic plot ------------------
def create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, categories,
                            output_path, ncol_legend=4, display_labels=None):
    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(metric_labels))
    cats = categories
    lab = display_labels if display_labels is not None else cats

    total_slots = max(1, len(cats)) * 2
    slot_width = 0.8 / total_slots

    handles, leglabels = [], []
    for model_idx, (name, means, cis, cmap) in enumerate([
        ("GPT", gpt_means, gpt_cis, cmap_gpt),
        ("Claude", claude_means, claude_cis, cmap_claude)
    ]):
        for i, _ in enumerate(cats):
            slot = model_idx * len(cats) + i
            offset = (slot - (total_slots - 1)/2) * slot_width
            color = cmap(i / (len(cats)-1)) if len(cats) > 1 else cmap(0.5)
            bars = ax.bar(
                x + offset,
                [means[m][i] for m in range(len(metric_labels))],
                slot_width,
                yerr=[cis[m][i] for m in range(len(metric_labels))],
                capsize=ERR_CAPSIZE,
                color=color,
                label=f"{name} - {lab[i]}",
                alpha=0.9
            )
            handles.append(bars[0])
            leglabels.append(f"{name} - {lab[i]}")

    ax.set_ylabel("Score", fontsize=YLABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=XTICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ylim = _safe_ylim((gpt_means, gpt_cis), (claude_means, claude_cis))
    if ylim: ax.set_ylim(*ylim)

    finalize_and_save(fig, ax, output_path, handles, leglabels, ncol=ncol_legend,
                      legend_fontsize=LEGEND_FONTSIZE)

# ======================================================
# PLOT 1: GENDER
# ======================================================
if gender_col:
    gender_values = df[gender_col].dropna().astype(str).unique().tolist()
    # Prefer Male/Female if present, then anything else
    ordered = [g for g in ['male','female','Male','Female','M','F'] if g in gender_values]
    gender_order = ordered + [g for g in gender_values if g not in ordered]

    df_gpt_only = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for gender...")
    gpt_means, gpt_cis = means_cis_by_category(df_gpt_only, df_gpt_only[gender_col].astype(str), gender_order)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only[gender_col].astype(str), gender_order)

    create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, gender_order,
                            "outputs/figures/readability/by_gender.png", ncol_legend=min(4, len(gender_order)))

# ======================================================
# PLOT 2: EDUCATION (CSV-specific → HS / Univ / Med)
# ======================================================
if education_col:
    # These are exactly the three values in your CSVs
    # We'll keep this order if present
    preferred_order = [
        'high school diploma or lower',
        'university degree',
        'medical degree',
    ]
    present = df[education_col].dropna().astype(str).str.strip().unique().tolist()
    edu_order = [e for e in preferred_order if e in present] + [e for e in present if e not in preferred_order]

    # Abbreviated legend labels
    EDU_ABBR = {
        'high school diploma or lower': 'HS',
        'university degree': 'Univ',
        'medical degree': 'Med',
    }
    edu_labels_abbr = [EDU_ABBR.get(e, e) for e in edu_order]

    df_gpt_only = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for education...")
    gpt_means, gpt_cis = means_cis_by_category(df_gpt_only, df_gpt_only[education_col].astype(str), edu_order)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only[education_col].astype(str), edu_order)

    create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, edu_order,
                            "outputs/figures/readability/by_education.png",
                            ncol_legend=min(4, len(edu_order)),
                            display_labels=edu_labels_abbr)

# ======================================================
# PLOT 3: AGE GROUPS
# ======================================================
if age_col:
    bins = [-float('inf'), 17, 49, 64, float('inf')]
    age_labels = ['<18', '18–49', '50–64', '65+']
    df['AgeGroup'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'),
                            bins=bins, labels=age_labels, right=True)

    df_gpt_only = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for age groups...")
    gpt_means, gpt_cis = means_cis_by_category(df_gpt_only, df_gpt_only['AgeGroup'].astype(str), age_labels)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only['AgeGroup'].astype(str), age_labels)

    create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, age_labels,
                            "outputs/figures/readability/by_agegroup.png", ncol_legend=min(4, len(age_labels)))

# ======================================================
# PLOT 4: MEDICAL CONDITION
# ======================================================
if cond_col:
    cond_order = []
    preferred = ['PanCan','CIHD','Obes','Alz']
    present = df['cond_abbr'].dropna().astype(str).unique().tolist()
    for p in preferred:
        if p in present:
            cond_order.append(p)
    cond_order += sorted([c for c in present if c not in cond_order])

    df_gpt_only = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for conditions...")
    gpt_means, gpt_cis = means_cis_by_category(df_gpt_only, df_gpt_only['cond_abbr'].astype(str), cond_order)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only['cond_abbr'].astype(str), cond_order)

    ncols = 3 if len(cond_order) > 6 else min(4, len(cond_order))
    create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, cond_order,
                            "outputs/figures/readability/by_condition.png", ncol_legend=ncols)

# ======================================================
# PLOT 5: RACE/ETHNICITY
# ======================================================
if race_col:
    race_values = df[race_col].dropna().astype(str).unique().tolist()
    race_order = sorted(race_values)

    df_gpt_only = df[df["model"] == "GPT"]
    df_claude_only = df[df["model"] == "Claude"]

    print("Calculating bootstrap CIs for race/ethnicity...")
    gpt_means, gpt_cis = means_cis_by_category(df_gpt_only, df_gpt_only[race_col].astype(str), race_order)
    claude_means, claude_cis = means_cis_by_category(df_claude_only, df_claude_only[race_col].astype(str), race_order)

    ncol = min(len(race_order), 3)
    create_demographic_plot(gpt_means, gpt_cis, claude_means, claude_cis, race_order,
                            "outputs/figures/readability/by_race.png", ncol_legend=ncol)

# ======================================================
# (Optional) STATS: Paired tests by group
# ======================================================
def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    if np.any(passed):
        kmax = np.max(np.where(passed)[0]) + 1
        cutoff = ranked[kmax-1]
        reject = pvals <= cutoff
    else:
        reject = np.zeros_like(pvals, dtype=bool)
    adj = np.empty_like(ranked)
    cummin = 1.0
    for i in range(n-1, -1, -1):
        adj[i] = min(cummin, ranked[i] * n / (i+1))
        cummin = adj[i]
    p_adj = np.empty_like(pvals)
    p_adj[order] = np.minimum(adj, 1.0)
    return reject, p_adj

def paired_tests(df_gpt, df_claude, group_col, group_levels, label_name):
    if id_col is None:
        raise RuntimeError("No prompt ID column found to pair GPT/Claude.")
    results = []
    for level in group_levels:
        g_grp = df_gpt[df_gpt[group_col].astype(str) == str(level)]
        c_grp = df_claude[df_claude[group_col].astype(str) == str(level)]
        merged = pd.merge(
            g_grp[[id_col] + metric_cols],
            c_grp[[id_col] + metric_cols],
            on=id_col,
            suffixes=('_gpt','_claude')
        )
        for mc, label in zip(metric_cols, metric_labels):
            x = pd.to_numeric(merged[f"{mc}_gpt"], errors='coerce')
            y = pd.to_numeric(merged[f"{mc}_claude"], errors='coerce')
            mask = ~(x.isna() | y.isna())
            d = (x[mask] - y[mask]).to_numpy()
            n_eff = d.size
            if n_eff < 2:
                mdiff = np.nan; t_stat = np.nan; p_val = np.nan
                mg = x[mask].mean() if n_eff else np.nan
                mc_ = y[mask].mean() if n_eff else np.nan
            else:
                mdiff = float(np.mean(d))
                sd = float(np.std(d, ddof=1))
                t_stat = mdiff / (sd / np.sqrt(n_eff)) if sd > 0 else np.inf
                # normal approx for 2-sided p
                from math import erf, sqrt
                z = abs(t_stat)
                p_val = 2 * (1 - 0.5*(1 + erf(z / sqrt(2))))
                mg = float(x[mask].mean()); mc_ = float(y[mask].mean())
            results.append({
                label_name: level, 'metric': label, 'n_pairs': int(n_eff),
                'mean_gpt': mg, 'mean_claude': mc_, 'mean_diff': mdiff,
                't_stat': t_stat, 'p_raw': p_val
            })
    res_df = pd.DataFrame(results)
    mask_valid = res_df['p_raw'].notna()
    if mask_valid.any():
        reject, p_adj = fdr_bh(res_df.loc[mask_valid, 'p_raw'].values, alpha=0.05)
        res_df.loc[mask_valid, 'p_adj'] = p_adj
        res_df['significant_fdr_0.05'] = False
        res_df.loc[mask_valid, 'significant_fdr_0.05'] = reject
    else:
        res_df['p_adj'] = np.nan
        res_df['significant_fdr_0.05'] = False
    return res_df

# Save all stats (same as before)
if cond_col:
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    cond_levels = [c for c in df['cond_abbr'].dropna().astype(str).unique().tolist()]
    cond_levels = [c for c in ['PanCan','CIHD','Obes','Alz'] if c in cond_levels] + \
                  [c for c in cond_levels if c not in ['PanCan','CIHD','Obes','Alz']]
    cond_stats = paired_tests(df_g, df_c, 'cond_abbr', cond_levels, 'condition')
    cond_stats_path = "data/results/statistical_tests/stats_readability_by_condition.csv"
    os.makedirs(os.path.dirname(cond_stats_path), exist_ok=True)
    cond_stats.to_csv(cond_stats_path, index=False)
    print(f"Saved stats: {cond_stats_path}")

if age_col:
    age_levels = ['<18', '18–49', '50–64', '65+']
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    if 'AgeGroup' not in df.columns:
        bins = [-float('inf'), 17, 49, 64, float('inf')]
        df['AgeGroup'] = pd.cut(pd.to_numeric(df[age_col], errors='coerce'),
                                bins=bins, labels=age_levels, right=True)
    age_stats = paired_tests(df_g, df_c, 'AgeGroup', age_levels, 'age_group')
    age_stats_path = "data/results/statistical_tests/stats_readability_by_agegroup.csv"
    os.makedirs(os.path.dirname(age_stats_path), exist_ok=True)
    age_stats.to_csv(age_stats_path, index=False)
    print(f"Saved stats: {age_stats_path}")

if gender_col:
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    gender_levels = df[gender_col].dropna().astype(str).unique().tolist()
    gender_stats = paired_tests(df_g, df_c, gender_col, gender_levels, 'gender')
    gender_stats_path = "data/results/statistical_tests/stats_readability_by_gender.csv"
    os.makedirs(os.path.dirname(gender_stats_path), exist_ok=True)
    gender_stats.to_csv(gender_stats_path, index=False)
    print(f"Saved stats: {gender_stats_path}")

if education_col:
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    edu_levels = df[education_col].dropna().astype(str).unique().tolist()
    edu_stats = paired_tests(df_g, df_c, education_col, edu_levels, 'education')
    edu_stats_path = "data/results/statistical_tests/stats_readability_by_education.csv"
    os.makedirs(os.path.dirname(edu_stats_path), exist_ok=True)
    edu_stats.to_csv(edu_stats_path, index=False)
    print(f"Saved stats: {edu_stats_path}")

if race_col:
    df_g = df[df['model']=='GPT'].copy()
    df_c = df[df['model']=='Claude'].copy()
    race_levels = df[race_col].dropna().astype(str).unique().tolist()
    race_stats = paired_tests(df_g, df_c, race_col, race_levels, 'race')
    race_stats_path = "data/results/statistical_tests/stats_readability_by_race.csv"
    os.makedirs(os.path.dirname(race_stats_path), exist_ok=True)
    race_stats.to_csv(race_stats_path, index=False)
    print(f"Saved stats: {race_stats_path}")

print("\n" + "="*60)
print("✅ ALL PLOTS GENERATED WITH UNIFIED FONTS & EDUCATION ABBREVIATIONS (HS/Univ/Med)")
print("="*60)
print("Plots created:")
print("  1. outputs/figures/readability/overall.png")
print("  2. outputs/figures/readability/by_gender.png")
print("  3. outputs/figures/readability/by_education.png")
print("  4. outputs/figures/readability/by_agegroup.png")
print("  5. outputs/figures/readability/by_condition.png")
print("  6. outputs/figures/readability/by_race.png")
print("="*60)
