import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
gpt_df = pd.read_csv('gpt_response_gpt_rating.csv')
claude_df = pd.read_csv('gpt_response_claude_rating.csv')

# Define a function to process score distributions
def get_score_distributions(df, score_range=(1,5)):
    scores = np.arange(score_range[0], score_range[1]+1)
    
    empathy = df['Empathy Score'].value_counts()
    empathy = empathy.reindex(scores, fill_value=0).sort_index()
    
    understand = df['Understandability Score'].value_counts()
    understand = understand.reindex(scores, fill_value=0).sort_index()
    
    return empathy, understand

# Get distributions for both models
gpt_empathy, gpt_understand = get_score_distributions(gpt_df)
claude_empathy, claude_understand = get_score_distributions(claude_df)

# Create figure with subplots (increase figure size)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Set global font sizes
plt.rcParams.update({
    'axes.titlesize': 26,    # Title font size
    'axes.labelsize': 24,    # Axis label font size
    'xtick.labelsize': 22,   # X-axis tick label size
    'ytick.labelsize': 22,   # Y-axis tick label size
    'legend.fontsize': 22    # Legend font size
})

# Common parameters
bar_width = 0.35
x = np.arange(len(gpt_empathy))

# Plot Empathy Score comparison
ax1.bar(x - bar_width/2, gpt_empathy, bar_width, 
        label='GPT', color='#1f77b4', edgecolor='black')
ax1.bar(x + bar_width/2, claude_empathy, bar_width, 
        label='Claude', color='#ff7f0e', edgecolor='black')
ax1.set_title('Empathy Score Distribution', pad=20)
ax1.set_xlabel('Score')
ax1.set_ylabel('Count')
ax1.set_xticks(x)
ax1.set_xticklabels(gpt_empathy.index)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot Understandability Score comparison
ax2.bar(x - bar_width/2, gpt_understand, bar_width, 
        label='GPT', color='#1f77b4', edgecolor='black')
ax2.bar(x + bar_width/2, claude_understand, bar_width, 
        label='Claude', color='#ff7f0e', edgecolor='black')
ax2.set_title('Understandability Score Distribution', pad=20)
ax2.set_xlabel('Score')
ax2.set_ylabel('Count')
ax2.set_xticks(x)
ax2.set_xticklabels(gpt_understand.index)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()