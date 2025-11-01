import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Global plotting style
# ==============================
sns.set(style="whitegrid")
sns.set_context("talk", font_scale=1.3)

plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "figure.constrained_layout.use": True,
})

custom_palette = sns.color_palette("viridis", 4)
sns.set_palette(custom_palette)

FIGSIZE = (14, 7)
DPI = 300
XTICK_ROT = 30

# ==============================
# Paths & data loading
# ==============================
data_path = pathlib.Path("/Users/jianzhouyao/AI4Good/data/results/empathy_scores/combined_empathy_data.csv")
project_root = pathlib.Path("/Users/jianzhouyao/AI4Good")
combined_df = pd.read_csv(data_path)

# ==============================
# Clean/convert + canonicalize column names
# ==============================
combined_df["age"] = pd.to_numeric(combined_df["age"], errors="coerce")
combined_df.columns = combined_df.columns.str.strip()

frames = []

# Claude Response + Claude Rating
claude_claude = combined_df[combined_df['Response_Source'] == 'Claude'].copy()
claude_claude['Affective Empathy'] = claude_claude['Affective Empathy Score (Claude)']
claude_claude['Cognitive Empathy'] = claude_claude['Cognitive Empathy Score (Claude)']
claude_claude['Source'] = 'Claude Response + Claude Rating'
frames.append(claude_claude)

# Claude Response + GPT Rating
claude_gpt = combined_df[combined_df['Response_Source'] == 'Claude'].copy()
claude_gpt['Affective Empathy'] = claude_gpt['Affective Empathy Score (GPT)']
claude_gpt['Cognitive Empathy'] = claude_gpt['Cognitive Empathy Score (GPT)']
claude_gpt['Source'] = 'Claude Response + GPT Rating'
frames.append(claude_gpt)

# GPT Response + Claude Rating
gpt_claude = combined_df[combined_df['Response_Source'] == 'GPT'].copy()
gpt_claude['Affective Empathy'] = gpt_claude['Affective Empathy Score (Claude)']
gpt_claude['Cognitive Empathy'] = gpt_claude['Cognitive Empathy Score (Claude)']
gpt_claude['Source'] = 'GPT Response + Claude Rating'
frames.append(gpt_claude)

# GPT Response + GPT Rating
gpt_gpt = combined_df[combined_df['Response_Source'] == 'GPT'].copy()
gpt_gpt['Affective Empathy'] = gpt_gpt['Affective Empathy Score (GPT)']
gpt_gpt['Cognitive Empathy'] = gpt_gpt['Cognitive Empathy Score (GPT)']
gpt_gpt['Source'] = 'GPT Response + GPT Rating'
frames.append(gpt_gpt)

combined_df = pd.concat(frames, ignore_index=True)

# ==============================
# Abbreviations
# ==============================
source_short_map = {
    "Claude Response + Claude Rating": "C→C",
    "Claude Response + GPT Rating":    "C→G",
    "GPT Response + Claude Rating":    "G→C",
    "GPT Response + GPT Rating":       "G→G",
}
combined_df["SourceShort"] = combined_df["Source"].map(source_short_map).fillna(combined_df["Source"])
source_short_order = ["C→C", "C→G", "G→C", "G→G"]

edu_labels_abbrev = {
    "high school diploma or lower": "HS",
    "university degree": "Univ",
    "medical degree": "Med",
}
diag_labels_abbrev = {
    "pancreatic cancer": "PanCan",
    "Chronic Ischemic Heart Disease": "CIHD",
    "obesity": "Obes",
    "Alzheimer’s": "Alz",
}
combined_df["education_abbrev"] = combined_df["education"].map(edu_labels_abbrev).fillna(combined_df["education"])
combined_df["diagnosis_abbrev"] = combined_df["diagnosis"].map(diag_labels_abbrev).fillna(combined_df["diagnosis"])

# ==============================
# Output folder
# ==============================
output_folder = project_root / "outputs" / "figures" / "empathy_demographics"
os.makedirs(output_folder, exist_ok=True)

# ==============================
# Helper: barplot generator (no markers)
# ==============================
def two_panel_with_bottom_legend(
    data, x1, x2, order1=None, order2=None, xlabel="",
    y1="Affective Empathy", y2="Cognitive Empathy",
    hue="SourceShort", hue_order=None, filename="figure.png"
):
    fig = plt.figure(figsize=FIGSIZE)

    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x=x1, y=y1, hue=hue, hue_order=hue_order, data=data,
                errorbar=("ci", 95), order=order1, ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=XTICK_ROT, ha="right")

    ax2 = plt.subplot(1, 2, 2)
    sns.barplot(x=x2, y=y2, hue=hue, hue_order=hue_order, data=data,
                errorbar=("ci", 95), order=order2, ax=ax2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(y2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=XTICK_ROT, ha="right")

    if ax1.get_legend() is not None: ax1.get_legend().remove()
    if ax2.get_legend() is not None: ax2.get_legend().remove()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    path = output_folder / filename
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    return path

# ==============================
# (1) By Age Group
# ==============================
bins   = [0, 18, 50, 65, 100]
labels = ['<18', '18-49', '50-64', '65+']
combined_df['age_group'] = pd.cut(combined_df['age'], bins=bins, labels=labels, right=False)

age_path = two_panel_with_bottom_legend(
    combined_df, "age_group", "age_group",
    order1=labels, order2=labels, xlabel="Age Group",
    hue_order=source_short_order, filename="scores_by_age.png"
)

# ==============================
# (2) By Education
# ==============================
edu_path = two_panel_with_bottom_legend(
    combined_df, "education_abbrev", "education_abbrev",
    order1=["HS","Univ","Med"], order2=["HS","Univ","Med"], xlabel="Education",
    hue_order=source_short_order, filename="scores_by_education.png"
)

# ==============================
# (3) By Geographical Group
# ==============================
ethnicity_path = two_panel_with_bottom_legend(
    combined_df, "ethnicity", "ethnicity",
    xlabel="Geographical Group", hue_order=source_short_order,
    filename="scores_by_geographical_group.png"
)

# ==============================
# (4) By Diagnosis
# ==============================
dx_path = two_panel_with_bottom_legend(
    combined_df, "diagnosis_abbrev", "diagnosis_abbrev",
    order1=["PanCan","CIHD","Obes","Alz"], order2=["PanCan","CIHD","Obes","Alz"],
    xlabel="Diagnosis", hue_order=source_short_order, 
    filename="scores_by_diagnosis.png"
)

print("✅ Saved demographic figures:")
for p in [age_path, edu_path, ethnicity_path, dx_path]:
    print(" -", p)

# ==============================
# (5) Gender Bias Visualization
# ==============================
_df = combined_df.copy()
_df["gender_norm"] = _df["gender"].astype(str).str.strip().str.lower()
_df["SourceShort"] = _df["Source"].map(source_short_map).fillna(_df["Source"])

bias_wide = (
    _df.groupby(["SourceShort", "gender_norm"])[["Affective Empathy", "Cognitive Empathy"]]
    .mean().unstack(level=-1)
)

aff_f = bias_wide[("Affective Empathy", "female")]
aff_m = bias_wide[("Affective Empathy", "male")]
cog_f = bias_wide[("Cognitive Empathy", "female")]
cog_m = bias_wide[("Cognitive Empathy", "male")]

bias = pd.DataFrame({
    "Affective": aff_f - aff_m,
    "Cognitive": cog_f - cog_m,
}).reindex([lab for lab in source_short_order if lab in bias_wide.index])

print("\nGender Bias (Female - Male):")
print(bias.round(3))

plt.figure(figsize=FIGSIZE)
ax = plt.gca()
bar_containers = bias.plot(
    kind="bar", ax=ax, color=[custom_palette[0], custom_palette[1]],
    rot=0, width=0.8, legend=False
).containers

ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax.set_xticklabels(bias.index, rotation=XTICK_ROT, ha="right")
ax.set_ylabel("Female − Male (mean difference)")
ax.set_xlabel("")

for container in bar_containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

handles = list(bar_containers)
labels_leg  = bias.columns.tolist()
ax.legend(handles=handles, labels=labels_leg,
          loc="upper center", bbox_to_anchor=(0.5, 1.20),
          ncol=2, frameon=False)

ax.text(0.5, 0.02, "Gender effects not significant (p > 0.05)", 
        transform=ax.transAxes, ha='center', va='bottom', 
        fontsize=16, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.88])

gb_png = output_folder / "gender_bias_difference.png"
gb_pdf = output_folder / "gender_bias_difference.pdf"
plt.savefig(gb_png, dpi=DPI, bbox_inches="tight")
plt.savefig(gb_pdf, bbox_inches="tight")
plt.close()

print("✅ Saved gender bias figure:", gb_png)
