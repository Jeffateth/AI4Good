# =============================================================================
# READABILITY METRICS — MEAN ± 95% CI (viridis style, same layout)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as st
from pathlib import Path

# --- Global style (same as your empathy plots) ---
sns.set(style="whitegrid")
sns.set_context("talk", font_scale=1.3)

plt.rcParams.update({
    "font.size": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 22,
    "figure.constrained_layout.use": True,
})

custom_palette = sns.color_palette("viridis", 4)
sns.set_palette(custom_palette)

FIGSIZE = (10, 6)
DPI = 300
XTICK_ROT = 15

# Ensure output folder exists
output_folder = Path("outputs/figures/readability")
output_folder.mkdir(parents=True, exist_ok=True)

# Helper: compute mean ± 95% CI (t-based)
def mean_ci(values, confidence=0.95):
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    sem = st.sem(arr)
    h = sem * st.t.ppf((1 + confidence) / 2., len(arr) - 1)
    return mean, h

# Helper: place legend outside
def place_legend_outside(ax, title=None, loc="upper left"):
    ax.legend(title=title, bbox_to_anchor=(1.02, 1), loc=loc, borderaxespad=0)

width = 0.35

# Colors from viridis palette
color_claude = custom_palette[1]
color_gpt    = custom_palette[2]

# =============================================================================
# Example data: replace each list with your per-response readability scores
# =============================================================================
# These are placeholder arrays — replace with your real data arrays
# e.g., Flesch–Kincaid scores for 156 GPT responses etc.
flesch_gpt   = np.random.normal(9.72, 0.6, 156)
flesch_cla   = np.random.normal(9.54, 0.6, 156)
smog_gpt     = np.random.normal(12.59, 0.5, 156)
smog_cla     = np.random.normal(12.13, 0.5, 156)
fog_gpt      = np.random.normal(11.86, 0.5, 156)
fog_cla      = np.random.normal(11.70, 0.5, 156)

# =============================================================================
# Plot 1: Overall Metric Comparison
# =============================================================================
metrics = ["Flesch-Kincaid", "SMOG Index", "Gunning Fog"]
gpt_data = [flesch_gpt, smog_gpt, fog_gpt]
cla_data = [flesch_cla, smog_cla, fog_cla]

gpt_means, gpt_errs = zip(*[mean_ci(d) for d in gpt_data])
cla_means, cla_errs = zip(*[mean_ci(d) for d in cla_data])

scale_limits = [12, 18, 20]
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=FIGSIZE)

bars_cla = ax.bar(x - width/2, cla_means, width, yerr=cla_errs, capsize=5,
                  label='Claude', color=color_claude)
bars_gpt = ax.bar(x + width/2, gpt_means, width, yerr=gpt_errs, capsize=5,
                  label='GPT', color=color_gpt)

def draw_horizontal_stripes(ax, x_center, y_level, total_width, n_stripes=7, stripe_len_frac=0.25):
    spacing = total_width / n_stripes
    start_x = x_center - total_width / 2
    stripe_len = stripe_len_frac * spacing
    for i in range(n_stripes):
        x_start = start_x + i * spacing
        x_end = x_start + stripe_len
        ax.hlines(y=y_level, xmin=x_start, xmax=x_end, color='black', linewidth=1)

for i, limit in enumerate(scale_limits):
    draw_horizontal_stripes(ax, x[i] - width/2, limit, width)
    draw_horizontal_stripes(ax, x[i] + width/2, limit, width)

# Value labels
for bars in [bars_cla, bars_gpt]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

ax.set_ylabel('Grade Level')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=XTICK_ROT, ha='right')

scale_line = Line2D([0], [0], color='black', linewidth=1, label='Scale Limit')
handles = [bars_cla, bars_gpt, scale_line]
labels  = ['Claude', 'GPT', 'Scale Limit']
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1),
          loc="upper left", borderaxespad=0)

plt.tight_layout()
plt.savefig(output_folder / "readability_overall_metrics.png", dpi=DPI, bbox_inches="tight")
plt.close()

# =============================================================================
# Plot 2: Education Level — Flesch-Kincaid
# =============================================================================
edu_levels = ["High school", "University", "Medical"]
gpt_edu = [np.random.normal(8.29, 0.5, 52),
           np.random.normal(10.03, 0.5, 52),
           np.random.normal(11.19, 0.5, 52)]
cla_edu = [np.random.normal(6.81, 0.5, 52),
           np.random.normal(10.44, 0.5, 52),
           np.random.normal(12.05, 0.5, 52)]

gpt_means, gpt_errs = zip(*[mean_ci(d) for d in gpt_edu])
cla_means, cla_errs = zip(*[mean_ci(d) for d in cla_edu])

x = np.arange(len(edu_levels))
fig, ax = plt.subplots(figsize=FIGSIZE)

bars_gpt = ax.bar(x - width/2, gpt_means, width, yerr=gpt_errs, capsize=5,
                  label='GPT', color=color_gpt)
bars_cla = ax.bar(x + width/2, cla_means, width, yerr=cla_errs, capsize=5,
                  label='Claude', color=color_claude)

for bars in [bars_gpt, bars_cla]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha='center', va='bottom', fontsize=14)

ax.set_ylabel('Grade Level')
ax.set_ylim(0, 14)
ax.set_xticks(x)
ax.set_xticklabels(edu_levels, rotation=XTICK_ROT, ha='right')
place_legend_outside(ax)
plt.tight_layout()
plt.savefig(output_folder / "readability_by_education_fk.png", dpi=DPI, bbox_inches="tight")
plt.close()

# =============================================================================
# Plot 3: Ethnicity — SMOG Index
# =============================================================================
ethnicities = ["European", "African", "Asian"]
gpt_eth = [np.random.normal(12.76, 0.4, 52),
           np.random.normal(12.37, 0.4, 52),
           np.random.normal(12.63, 0.4, 52)]
cla_eth = [np.random.normal(12.39, 0.4, 52),
           np.random.normal(12.03, 0.4, 52),
           np.random.normal(11.97, 0.4, 52)]

gpt_means, gpt_errs = zip(*[mean_ci(d) for d in gpt_eth])
cla_means, cla_errs = zip(*[mean_ci(d) for d in cla_eth])

avg_eth = [np.mean([g, c]) for g, c in zip(gpt_means, cla_means)]
sorted_idx = np.argsort(avg_eth)
eth_sorted = [ethnicities[i] for i in sorted_idx]
gpt_sorted_means = [gpt_means[i] for i in sorted_idx]
gpt_sorted_errs  = [gpt_errs[i]  for i in sorted_idx]
cla_sorted_means = [cla_means[i] for i in sorted_idx]
cla_sorted_errs  = [cla_errs[i]  for i in sorted_idx]

x = np.arange(len(eth_sorted))
fig, ax = plt.subplots(figsize=FIGSIZE)

bars_gpt = ax.bar(x - width/2, gpt_sorted_means, width, yerr=gpt_sorted_errs, capsize=5,
                  label='GPT', color=color_gpt)
bars_cla = ax.bar(x + width/2, cla_sorted_means, width, yerr=cla_sorted_errs, capsize=5,
                  label='Claude', color=color_claude)

for bars in [bars_gpt, bars_cla]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha='center', va='bottom', fontsize=14)

ax.set_ylabel('SMOG Index')
ax.set_ylim(0, 14)
ax.set_xticks(x)
ax.set_xticklabels(eth_sorted, rotation=XTICK_ROT, ha='right')
place_legend_outside(ax)
plt.tight_layout()
plt.savefig(output_folder / "readability_by_ethnicity_smog.png", dpi=DPI, bbox_inches="tight")
plt.close()

# =============================================================================
# Plot 4: Gender — Gunning Fog
# =============================================================================
genders = ["Female", "Male"]
gpt_gen = [np.random.normal(11.99, 0.5, 78),
           np.random.normal(11.73, 0.5, 78)]
cla_gen = [np.random.normal(11.61, 0.5, 78),
           np.random.normal(11.78, 0.5, 78)]

gpt_means, gpt_errs = zip(*[mean_ci(d) for d in gpt_gen])
cla_means, cla_errs = zip(*[mean_ci(d) for d in cla_gen])

x = np.arange(len(genders))
fig, ax = plt.subplots(figsize=FIGSIZE)

bars_gpt = ax.bar(x - width/2, gpt_means, width, yerr=gpt_errs, capsize=5,
                  label='GPT', color=color_gpt)
bars_cla = ax.bar(x + width/2, cla_means, width, yerr=cla_errs, capsize=5,
                  label='Claude', color=color_claude)

for bars in [bars_gpt, bars_cla]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha='center', va='bottom', fontsize=14)

ax.set_ylabel('Gunning Fog')
ax.set_ylim(0, 14)
ax.set_xticks(x)
ax.set_xticklabels(genders, rotation=XTICK_ROT, ha='right')
place_legend_outside(ax, loc="upper left")
plt.tight_layout()
plt.savefig(output_folder / "readability_by_gender_gf.png", dpi=DPI, bbox_inches="tight")
plt.close()

print("✅ Saved 4 readability figures with 95% CIs to:", output_folder)
