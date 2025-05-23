import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def load_and_plot(path1, path2):

    df_human = pd.read_csv(path1)
    df_model = pd.read_csv(path2)
    

    aff_cols = [c for c in df_human.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_human.columns if c.startswith('Cognitive_Empathy_Human_')]
    

    df_human['Affective_human_mean'] = df_human[aff_cols].mean(axis=1)
    df_human['Affective_human_std']  = df_human[aff_cols].std(axis=1).fillna(0)
    df_human['Cognitive_human_mean'] = df_human[cog_cols].mean(axis=1)
    df_human['Cognitive_human_std']  = df_human[cog_cols].std(axis=1).fillna(0)
    
    
    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender']
    df = pd.merge(
        df_human[['Prompt Number','age','ethnicity','gender',
                  'Affective_human_mean','Affective_human_std',
                  'Cognitive_human_mean','Cognitive_human_std']],
        df_model,
        on=merge_cols
    )
    
    df = df[df['education'].str.lower() == 'high school diploma or lower']
    df['Category'] = df['ethnicity'] + ' ' + df['gender']
    categories = ['European female', 'African female', 'European male', 'African male']
    
    agg = df.groupby('Category').agg({
        'Affective_human_mean': 'mean',
        'Affective_human_std' : 'mean',
        'Affective Empathy Score (GPT)': 'mean',
        'Affective Empathy Score (Claude)': 'mean',
        'Cognitive_human_mean': 'mean',
        'Cognitive_human_std' : 'mean',
        'Cognitive Empathy Score (GPT)': 'mean',
        'Cognitive Empathy Score (Claude)': 'mean',
    }).reindex(categories)
    
    x = np.arange(len(categories))
    width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax0 = axs[0]
    ax0.bar(x - width, agg['Affective_human_mean'], width,
            yerr=agg['Affective_human_std'], capsize=5,
            label='Human Mean ± SD')
    ax0.bar(x,       agg['Affective Empathy Score (GPT)'], width, label='GPT')
    ax0.bar(x + width, agg['Affective Empathy Score (Claude)'], width, label='Claude')
    ax0.set_xticks(x)
    ax0.set_xticklabels(categories, rotation=15)
    ax0.set_title('Affective Empathy')
    ax0.set_ylabel('Score')
    ax0.legend()

    ax1 = axs[1]
    ax1.bar(x - width, agg['Cognitive_human_mean'], width,
            yerr=agg['Cognitive_human_std'], capsize=5,
            label='Human Mean ± SD')
    ax1.bar(x,       agg['Cognitive Empathy Score (GPT)'], width, label='GPT')
    ax1.bar(x + width, agg['Cognitive Empathy Score (Claude)'], width, label='Claude')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15)
    ax1.set_title('Cognitive Empathy')
    ax1.legend()

    plt.tight_layout()
    plt.show()



    df['GPT_diff_aff']    = df['Affective Empathy Score (GPT)']    - df['Affective_human_mean']
    df['Claude_diff_aff'] = df['Affective Empathy Score (Claude)'] - df['Affective_human_mean']
    df['GPT_diff_cog']    = df['Cognitive Empathy Score (GPT)']    - df['Cognitive_human_mean']
    df['Claude_diff_cog'] = df['Cognitive Empathy Score (Claude)'] - df['Cognitive_human_mean']

    
    df_scatter = df[df['Category'].isin(['European female', 'African female'])].copy()
    print(len(df_scatter))
 
    color_map = {
        'European female': 'C0',
        'African female':  'C1',
    }
    df_scatter['plot_color'] = df_scatter['Category'].map(color_map)

    category_patches = [
    mpatches.Patch(color=color_map['European female'], label='European female'),
    mpatches.Patch(color=color_map['African female'],  label='African female')
]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    ax = axs[0]
    sc_gpt_aff = ax.scatter(
        df_scatter['Affective_human_std'],
        df_scatter['GPT_diff_aff'],
        c=df_scatter['plot_color'],
        marker='o',
        s=150,                
        alpha=0.7,
        label='GPT'
    )
    sc_claude_aff = ax.scatter(
        df_scatter['Affective_human_std'],
        df_scatter['Claude_diff_aff'],
        c=df_scatter['plot_color'],
        marker='x',
        s=150,                
        alpha=0.7,
        label='Claude'
    )
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Human Affective SD')
    ax.set_ylabel('Model – Human Mean')
    ax.set_title('Affective Bias vs Human Dispersion')
    ax.legend(handles=[sc_gpt_aff, sc_claude_aff] + category_patches,
              loc='upper left', title='Legend')

    
    ax = axs[1]
    sc_gpt_cog = ax.scatter(
        df_scatter['Cognitive_human_std'],
        df_scatter['GPT_diff_cog'],
        c=df_scatter['plot_color'],
        marker='o',
        s=150,                
        alpha=0.7,
        label='GPT'
    )
    sc_claude_cog = ax.scatter(
        df_scatter['Cognitive_human_std'],
        df_scatter['Claude_diff_cog'],
        c=df_scatter['plot_color'],
        marker='x',
        s=150,               
        alpha=0.7,
        label='Claude'
    )
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Human Cognitive SD')
    ax.set_ylabel('Model – Human Mean')
    ax.set_title('Cognitive Bias vs Human Dispersion')
    ax.legend(handles=[sc_gpt_cog, sc_claude_cog] + category_patches,
              loc='upper left', title='Legend')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file1 = r'./initial_prompts_with_responses_gpt_human_final.csv'
    file2 = r'./gpt_response_with_ratings_updated.csv'
    load_and_plot(file1, file2)