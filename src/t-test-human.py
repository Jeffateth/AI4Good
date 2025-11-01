import os
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

RESULT_DIR = "/Users/jianzhouyao/AI4Good/data/results/statistical_tests"
os.makedirs(RESULT_DIR, exist_ok=True)


def run_category_tests(path1, path2):
    """
    Paired t-tests within each demographic category (education ≤ high school).
    Compares human mean scores vs GPT and Claude within each category.
    Returns DataFrame with raw results only (no BH correction).
    """
    df_h = pd.read_csv(path1)
    df_m = pd.read_csv(path2)

    # Compute mean human affective/cognitive empathy
    aff_cols = [c for c in df_h.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_h.columns if c.startswith('Cognitive_Empathy_Human_')]

    df_h['A_h_mean'] = df_h[aff_cols].mean(axis=1)
    df_h['C_h_mean'] = df_h[cog_cols].mean(axis=1)

    # Merge datasets
    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender', 'education']
    df = pd.merge(
        df_h[['Prompt Number', 'age', 'ethnicity', 'gender', 'education', 'A_h_mean', 'C_h_mean']],
        df_m,
        on=merge_cols
    )

    # Restrict to education ≤ high school
    df = df[df['education'].str.lower() == 'high school diploma or lower']

    # Build combined demographic categories
    df['Category'] = df['ethnicity'] + ' ' + df['gender']
    cats = ['European female', 'African female', 'European male', 'African male']
    df = df[df['Category'].isin(cats)].copy()

    results = []

    for cat in cats:
        sub = df[df['Category'] == cat].dropna()
        n = len(sub)
        if n < 2:
            continue

        tests = {
            'Affective human vs GPT': (sub['A_h_mean'], sub['Affective Empathy Score (GPT)']),
            'Affective human vs Claude': (sub['A_h_mean'], sub['Affective Empathy Score (Claude)']),
            'Cognitive human vs GPT': (sub['C_h_mean'], sub['Cognitive Empathy Score (GPT)']),
            'Cognitive human vs Claude': (sub['C_h_mean'], sub['Cognitive Empathy Score (Claude)']),
        }

        for label, (a, b) in tests.items():
            t, p = ttest_rel(a, b)
            results.append({
                "TestType": "Paired (within category)",
                "Category": cat,
                "Comparison": label,
                "n": n,
                "t_value": t,
                "p_value": p,
                "Significant": p < 0.05
            })

    return pd.DataFrame(results)


def compare_groups(path1, path2):
    """
    Two-sample t-tests comparing African female vs European female.
    Returns DataFrame with raw results only (no BH correction).
    """
    df_h = pd.read_csv(path1)
    df_m = pd.read_csv(path2)

    aff_cols = [c for c in df_h.columns if c.startswith('Affective_Empathy_Human_')]
    cog_cols = [c for c in df_h.columns if c.startswith('Cognitive_Empathy_Human_')]

    df_h['A_h_mean'] = df_h[aff_cols].mean(axis=1)
    df_h['C_h_mean'] = df_h[cog_cols].mean(axis=1)

    merge_cols = ['Prompt Number', 'age', 'ethnicity', 'gender', 'education']
    df = pd.merge(
        df_h[['Prompt Number', 'age', 'ethnicity', 'gender', 'education', 'A_h_mean', 'C_h_mean']],
        df_m,
        on=merge_cols
    )

    df = df[df['education'].str.lower() == 'high school diploma or lower']

    g1 = (df['ethnicity'] == 'African') & (df['gender'] == 'female')
    g2 = (df['ethnicity'] == 'European') & (df['gender'] == 'female')

    comparisons = {
        'Affective human': 'A_h_mean',
        'Affective GPT': 'Affective Empathy Score (GPT)',
        'Affective Claude': 'Affective Empathy Score (Claude)',
        'Cognitive human': 'C_h_mean',
        'Cognitive GPT': 'Cognitive Empathy Score (GPT)',
        'Cognitive Claude': 'Cognitive Empathy Score (Claude)',
    }

    results = []
    for label, col in comparisons.items():
        s1 = df.loc[g1, col].dropna()
        s2 = df.loc[g2, col].dropna()
        if len(s1) > 1 and len(s2) > 1:
            t, p = ttest_ind(s1, s2, equal_var=False)
            results.append({
                "TestType": "Two-sample (between groups)",
                "Category": "African female vs European female",
                "Comparison": label,
                "n1": len(s1),
                "n2": len(s2),
                "t_value": t,
                "p_value": p,
                "Significant": p < 0.05
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    file1 = "/Users/jianzhouyao/AI4Good/data/processed/ratings/human_ratings.csv"
    file2 = "/Users/jianzhouyao/AI4Good/data/processed/ratings/gpt_with_ratings.csv"

    print("Running paired category tests...")
    df_cat = run_category_tests(file1, file2)
    print(f"→ Found {len(df_cat)} paired results")

    print("Running group comparison tests...")
    df_grp = compare_groups(file1, file2)
    print(f"→ Found {len(df_grp)} group results")

    # Combine and save both
    all_results = pd.concat([df_cat, df_grp], ignore_index=True)
    out_path = os.path.join(RESULT_DIR, "human_empathy_t_tests.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\n✅ All results saved to:\n{out_path}")
