import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind


def run_category_tests(path1, path2):
    
    df_h = pd.read_csv(path1)
    df_m = pd.read_csv(path2)

    
    aff_cols = [c for c in df_h.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_h.columns if c.startswith('Cognitive_Empathy_Human_')]

    
    df_h['A_h_mean'] = df_h[aff_cols].mean(axis=1)
    df_h['C_h_mean'] = df_h[cog_cols].mean(axis=1)

    
    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender', 'education']
    df = pd.merge(
        df_h[['Prompt Number','age','ethnicity','gender','education','A_h_mean','C_h_mean']],
        df_m,
        on=merge_cols
    )

    
    df = df[df['education'].str.lower() == 'high school diploma or lower']

    
    df['Category'] = df['ethnicity'] + ' ' + df['gender']
    cats = ['European female', 'African female', 'European male', 'African male']
    df = df[df['Category'].isin(cats)].copy()

    
    print("=== Paired t-test within each category (education ≤ high school) ===\n")
    for cat in cats:
        sub = df[df['Category'] == cat]
        n = len(sub)
        print(f">> Category: {cat} (n={n})")
        if n < 2:
            print("   Not enough samples for t-test (n < 2)\n")
            continue

        # Affective: human vs GPT
        t_aff_gpt, p_aff_gpt = ttest_rel(sub['A_h_mean'], sub['Affective Empathy Score (GPT)'])
        # Affective: human vs Claude
        t_aff_cl,  p_aff_cl  = ttest_rel(sub['A_h_mean'], sub['Affective Empathy Score (Claude)'])
        # Cognitive: human vs GPT
        t_cog_gpt, p_cog_gpt = ttest_rel(sub['C_h_mean'], sub['Cognitive Empathy Score (GPT)'])
        # Cognitive: human vs Claude
        t_cog_cl,  p_cog_cl  = ttest_rel(sub['C_h_mean'], sub['Cognitive Empathy Score (Claude)'])

        print(f"   Affective human vs GPT:    t={t_aff_gpt:.2f}, p={p_aff_gpt:.3f}")
        print(f"   Affective human vs Claude: t={t_aff_cl:.2f}, p={p_aff_cl:.3f}")
        print(f"   Cognitive human vs GPT:    t={t_cog_gpt:.2f}, p={p_cog_gpt:.3f}")
        print(f"   Cognitive human vs Claude: t={t_cog_cl:.2f}, p={p_cog_cl:.3f}\n")

def compare_groups(path1, path2):
    """
    Conducts two-sample t-tests between two demographic groups
    on affective and cognitive empathy for human, GPT, and Claude.
    """
    df_h = pd.read_csv(path1)
    df_m = pd.read_csv(path2)

    
    aff_cols = [c for c in df_h.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_h.columns if c.startswith('Cognitive_Empathy_Human_')]

    
    df_h['A_h_mean'] = df_h[aff_cols].mean(axis=1)
    df_h['C_h_mean'] = df_h[cog_cols].mean(axis=1)

    
    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender', 'education']
    df = pd.merge(
        df_h[['Prompt Number','age','ethnicity','gender','education','A_h_mean','C_h_mean']],
        df_m,
        on=merge_cols
    )

    df = df[df['education'].str.lower() == 'high school diploma or lower']

    
    g1 = (df['ethnicity'] == 'African') & (df['gender'] == 'female')
    g2 = (df['ethnicity'] == 'European') & (df['gender'] == 'female')

    comparisons = {
        'Affective human':    'A_h_mean',
        'Affective GPT':      'Affective Empathy Score (GPT)',
        'Affective Claude':   'Affective Empathy Score (Claude)',
        'Cognitive human':    'C_h_mean',
        'Cognitive GPT':      'Cognitive Empathy Score (GPT)',
        'Cognitive Claude':   'Cognitive Empathy Score (Claude)',
    }

    print("Two-sample t-test: African female vs European female (education ≤ high school)\n")
    for label, col in comparisons.items():
        s1 = df.loc[g1, col].dropna()
        s2 = df.loc[g2, col].dropna()
        if len(s1) > 1 and len(s2) > 1:
            t, p = ttest_ind(s1, s2, equal_var=False)
            print(f"{label:20s} t = {t:6.2f}, p = {p:.3f}")
        else:
            print(f"{label:20s} not enough data (n1={len(s1)}, n2={len(s2)})")


if __name__ == '__main__':
    file1 = r'./initial_prompts_with_responses_gpt_human_final.csv'
    file2 = r'./gpt_response_with_ratings_updated.csv'
    run_category_tests(file1, file2)
    # Example usage
    compare_groups(file1, file2)
