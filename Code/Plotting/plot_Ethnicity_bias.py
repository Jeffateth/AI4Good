import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Ratings/claude_response_with_ratings.csv")

# Columns to analyze
metrics = {
    "Affective Empathy": [
        "Affective Empathy Score (GPT)",
        "Affective Empathy Score (Claude)"
    ],
    "Cognitive Understanding": [
        "Cognitive Understanding Score (GPT)",
        "Cognitive Understanding Score (Claude)"
    ]
}

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for i, (title, cols) in enumerate(metrics.items()):
    avg_scores = df.groupby("ethnicity")[cols].mean()
    avg_scores.plot(kind='bar', ax=axes[i])
    axes[i].set_title(f"{title} by Ethnicity")
    axes[i].set_ylabel("Average Score (Likert scale 1–3)")
    axes[i].set_xlabel("Ethnicity")
    axes[i].set_ylim(1, 3)
    axes[i].legend(title="Model")

plt.suptitle("Claude-Generated Responses: Empathy Score Comparison by Ethnicity", fontsize=16, y=0.98)

plt.tight_layout()
plt.show()
