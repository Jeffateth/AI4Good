import pandas as pd
import matplotlib.pyplot as plt

# Load the two datasets
df1 = pd.read_csv("Ratings/claude_response_with_ratings.csv")
df2 = pd.read_csv("Ratings/gpt_response_with_ratings.csv")

# Columns to compare
metrics = [
    "Implicit Emotion Recognition",
    "Intention Recognition",
    "Key Event Recognition",
    "Mixed Event Recognition"
]

# Calculate mean for each metric
means_df1 = df1[metrics].mean()
means_df2 = df2[metrics].mean()

# Plotting
x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([p - width/2 for p in x], means_df1, width=width, label="claude response")
plt.bar([p + width/2 for p in x], means_df2, width=width, label="gpt response")

plt.xticks(ticks=x, labels=metrics, rotation=20)
plt.ylabel("Average Score")
plt.title("Comparison of Empathy Metrics Between Claude and GPT responses")
plt.legend()
plt.tight_layout()
plt.show()
