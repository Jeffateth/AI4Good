# PROPER EMPATHY ANALYSIS - With Benjamini-Hochberg Correction
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMPATHY ANALYSIS - WITH BENJAMINI-HOCHBERG CORRECTION")
print("=" * 80)

# Load the two files
claude_df = pd.read_csv('/Users/jianzhouyao/AI4Good/data/processed/ratings/claude_with_ratings.csv')
gpt_df = pd.read_csv('/Users/jianzhouyao/AI4Good/data/processed/ratings/gpt_with_ratings.csv')

print("📊 DATA OVERVIEW")
print("-" * 40)
print(f"Claude responses: {len(claude_df)} rows")
print(f"GPT responses: {len(gpt_df)} rows")
print(f"Total unique responses: {len(claude_df) + len(gpt_df)}")

# Add response source identifier
claude_df['Response_Source'] = 'Claude'
gpt_df['Response_Source'] = 'GPT'

# Create age groups in individual dataframes first
claude_df['age_group'] = pd.cut(claude_df['age'], 
                               bins=[0, 18, 50, 65, 100], 
                               labels=['<18', '18-49', '50-64', '65+'], 
                               right=False)

gpt_df['age_group'] = pd.cut(gpt_df['age'], 
                            bins=[0, 18, 50, 65, 100], 
                            labels=['<18', '18-49', '50-64', '65+'], 
                            right=False)

# Combine for demographic analysis
combined_df = pd.concat([claude_df, gpt_df], ignore_index=True)

print(f"\nPrompt overlap check:")
claude_prompts = set(claude_df['Prompt Number'])
gpt_prompts = set(gpt_df['Prompt Number'])
overlap = len(claude_prompts.intersection(gpt_prompts))
print(f"Shared prompt numbers: {overlap} (same demographic scenarios)")

print(f"\n📋 SCORE OVERVIEW")
print("-" * 40)
score_columns = [
    'Affective Empathy Score (GPT)', 
    'Affective Empathy Score (Claude)',
    'Cognitive Empathy Score (GPT)', 
    'Cognitive Empathy Score (Claude)'
]

for col in score_columns:
    scores = combined_df[col].dropna()
    print(f"{col}: {scores.min()}-{scores.max()}, mean={scores.mean():.2f}, n={len(scores)}")

# GLOBAL STRUCTURE TO COLLECT ALL P-VALUES FOR BH CORRECTION
all_tests = []

print("\n" + "=" * 80)
print("1. RATER AGREEMENT ANALYSIS")
print("=" * 80)

def analyze_rater_agreement(df, response_type):
    """Analyze agreement between Claude and GPT raters"""
    print(f"\n🤝 RATER AGREEMENT: {response_type} Responses")
    print("-" * 50)
    
    local_tests = []
    
    # Affective empathy agreement
    aff_gpt = df['Affective Empathy Score (GPT)'].dropna()
    aff_claude = df['Affective Empathy Score (Claude)'].dropna()
    
    if len(aff_gpt) > 0 and len(aff_claude) > 0:
        # Correlation
        aff_corr = np.corrcoef(aff_gpt, aff_claude)[0,1]
        print(f"Affective Empathy Correlation: r = {aff_corr:.3f}")
        
        # Paired t-test (systematic bias?)
        t_stat_aff, p_val_aff = ttest_rel(aff_claude, aff_gpt)
        mean_diff_aff = aff_claude.mean() - aff_gpt.mean()
        print(f"Claude vs GPT bias: {mean_diff_aff:+.3f}, p = {p_val_aff:.4f}")
        
        local_tests.append({
            'test': f'Rater Agreement: {response_type} Affective',
            'p_value': p_val_aff,
            'effect': mean_diff_aff,
            'test_type': 'paired_t'
        })
        
        if p_val_aff < 0.05:
            rater_direction = "Claude rates higher" if mean_diff_aff > 0 else "GPT rates higher"
            print(f"  ✓ SIGNIFICANT rater bias (uncorrected): {rater_direction}")
        else:
            print(f"  ✗ No significant rater bias")
    
    # Cognitive empathy agreement  
    cog_gpt = df['Cognitive Empathy Score (GPT)'].dropna()
    cog_claude = df['Cognitive Empathy Score (Claude)'].dropna()
    
    if len(cog_gpt) > 0 and len(cog_claude) > 0:
        cog_corr = np.corrcoef(cog_gpt, cog_claude)[0,1]
        print(f"Cognitive Empathy Correlation: r = {cog_corr:.3f}")
        
        t_stat_cog, p_val_cog = ttest_rel(cog_claude, cog_gpt)
        mean_diff_cog = cog_claude.mean() - cog_gpt.mean()
        print(f"Claude vs GPT bias: {mean_diff_cog:+.3f}, p = {p_val_cog:.4f}")
        
        local_tests.append({
            'test': f'Rater Agreement: {response_type} Cognitive',
            'p_value': p_val_cog,
            'effect': mean_diff_cog,
            'test_type': 'paired_t'
        })
        
        if p_val_cog < 0.05:
            rater_direction = "Claude rates higher" if mean_diff_cog > 0 else "GPT rates higher"
            print(f"  ✓ SIGNIFICANT rater bias (uncorrected): {rater_direction}")
        else:
            print(f"  ✗ No significant rater bias")
    
    return local_tests

# Analyze rater agreement for each response type
all_tests.extend(analyze_rater_agreement(claude_df, "Claude"))
all_tests.extend(analyze_rater_agreement(gpt_df, "GPT"))

print("\n" + "=" * 80)
print("2. DEMOGRAPHIC BIAS ANALYSIS")
print("=" * 80)

def analyze_demographic_patterns(df, response_type, rater_type):
    """Analyze demographic biases for a specific response-rater combination"""
    
    print(f"\n🎯 {response_type} Responses → {rater_type} Ratings")
    print("-" * 50)
    
    aff_col = f'Affective Empathy Score ({rater_type})'
    cog_col = f'Cognitive Empathy Score ({rater_type})'
    
    results = {}
    local_tests = []
    
    context = f"{response_type}→{rater_type}"
    
    # 1. GENDER ANALYSIS
    print(f"\n1. Gender Analysis")
    male_data = df[df['gender'] == 'male']
    female_data = df[df['gender'] == 'female']
    
    print(f"Sample sizes: Male={len(male_data)}, Female={len(female_data)}")
    
    if len(male_data) >= 2 and len(female_data) >= 2:
        # Affective empathy gender test
        male_aff = male_data[aff_col].dropna()
        female_aff = female_data[aff_col].dropna()
        
        if len(male_aff) >= 2 and len(female_aff) >= 2:
            t_stat, p_val = ttest_ind(female_aff, male_aff)
            bias = female_aff.mean() - male_aff.mean()
            
            # Effect size
            pooled_std = np.sqrt(((len(female_aff)-1)*female_aff.var() + (len(male_aff)-1)*male_aff.var()) / (len(female_aff)+len(male_aff)-2))
            cohens_d = bias / pooled_std if pooled_std > 0 else 0
            
            print(f"  Affective: Female-Male = {bias:+.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
            results['gender_aff_p'] = p_val
            results['gender_aff_d'] = cohens_d
            
            local_tests.append({
                'test': f'Affective: {context} Gender',
                'p_value': p_val,
                'effect': bias,
                'cohens_d': cohens_d,
                'test_type': 'independent_t'
            })
            
            if p_val < 0.05:
                print(f"    ✓ SIGNIFICANT gender bias (uncorrected)!")
            elif abs(cohens_d) >= 0.2:
                print(f"    ~ Meaningful effect size")
        
        # Cognitive empathy gender test
        male_cog = male_data[cog_col].dropna()
        female_cog = female_data[cog_col].dropna()
        
        if len(male_cog) >= 2 and len(female_cog) >= 2:
            t_stat, p_val = ttest_ind(female_cog, male_cog)
            bias = female_cog.mean() - male_cog.mean()
            
            pooled_std = np.sqrt(((len(female_cog)-1)*female_cog.var() + (len(male_cog)-1)*male_cog.var()) / (len(female_cog)+len(male_cog)-2))
            cohens_d = bias / pooled_std if pooled_std > 0 else 0
            
            print(f"  Cognitive: Female-Male = {bias:+.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
            results['gender_cog_p'] = p_val
            results['gender_cog_d'] = cohens_d
            
            local_tests.append({
                'test': f'Cognitive: {context} Gender',
                'p_value': p_val,
                'effect': bias,
                'cohens_d': cohens_d,
                'test_type': 'independent_t'
            })
            
            if p_val < 0.05:
                print(f"    ✓ SIGNIFICANT gender bias (uncorrected)!")
            elif abs(cohens_d) >= 0.2:
                print(f"    ~ Meaningful effect size")
    
    # 2. AGE GROUP ANALYSIS
    print(f"\n2. Age Group Analysis")
    age_stats = df.groupby('age_group')[aff_col].agg(['count', 'mean', 'std']).round(3)
    print("Age group statistics (Affective Empathy):")
    print(age_stats)
    
    # Test U-shaped pattern
    age_groups = ['<18', '18-49', '50-64', '65+']
    age_data = []
    age_means = []
    
    for age_group in age_groups:
        scores = df[df['age_group'] == age_group][aff_col].dropna()
        if len(scores) >= 2:
            age_data.append(scores)
            age_means.append(scores.mean())
        else:
            age_data.append(None)
            age_means.append(None)
    
    # Test overall age effect
    valid_age_data = [data for data in age_data if data is not None]
    if len(valid_age_data) >= 2:
        f_stat, p_val = f_oneway(*valid_age_data)
        print(f"Age ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
        results['age_anova_p'] = p_val
        
        local_tests.append({
            'test': f'Affective: {context} Age (ANOVA)',
            'p_value': p_val,
            'effect': f_stat,
            'test_type': 'anova'
        })
        
        # Test U-shape specifically (young + old vs middle)
        if age_data[0] is not None and age_data[3] is not None and age_data[1] is not None and age_data[2] is not None:
            young_old = pd.concat([age_data[0], age_data[3]])  # <18 + 65+
            middle = pd.concat([age_data[1], age_data[2]])     # 18-49 + 50-64
            
            t_stat, p_val_u = ttest_ind(young_old, middle)
            print(f"U-shape test: t={t_stat:.3f}, p={p_val_u:.4f}")
            results['age_u_shape_p'] = p_val_u
            
            local_tests.append({
                'test': f'Affective: {context} Age (U-shape)',
                'p_value': p_val_u,
                'effect': young_old.mean() - middle.mean(),
                'test_type': 'independent_t'
            })
            
            if p_val_u < 0.05:
                print(f"    ✓ SIGNIFICANT U-shaped age pattern (uncorrected)!")
            else:
                print(f"    ✗ U-shape not significant")
    
    # 3. ETHNICITY ANALYSIS
    print(f"\n3. Ethnicity Analysis")
    eth_stats = df.groupby('ethnicity')[aff_col].agg(['count', 'mean', 'std']).round(3)
    print("Ethnicity statistics (Affective Empathy):")
    print(eth_stats)
    
    # Test ethnicity differences
    ethnicities = df['ethnicity'].unique()
    eth_data = []
    eth_means = []
    
    for eth in ethnicities:
        scores = df[df['ethnicity'] == eth][aff_col].dropna()
        if len(scores) >= 2:
            eth_data.append(scores)
            eth_means.append(scores.mean())
            
    if len(eth_data) >= 2:
        f_stat, p_val = f_oneway(*eth_data)
        print(f"Ethnicity ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
        results['ethnicity_anova_p'] = p_val
        
        local_tests.append({
            'test': f'Affective: {context} Ethnicity',
            'p_value': p_val,
            'effect': f_stat,
            'test_type': 'anova'
        })
        
        if p_val < 0.05:
            print(f"    ✓ SIGNIFICANT ethnicity differences (uncorrected)!")
            # Find highest and lowest
            max_idx = np.argmax(eth_means)
            min_idx = np.argmin(eth_means)
            print(f"    Highest: {list(ethnicities)[max_idx]} ({eth_means[max_idx]:.3f})")
            print(f"    Lowest: {list(ethnicities)[min_idx]} ({eth_means[min_idx]:.3f})")
    
    # 4. EDUCATION ANALYSIS
    print(f"\n4. Education Analysis")
    edu_stats = df.groupby('education')[aff_col].agg(['count', 'mean', 'std']).round(3)
    print("Education statistics (Affective Empathy):")
    print(edu_stats)
    
    # Test education hierarchy
    education_levels = df['education'].unique()
    if 'high school diploma or lower' in education_levels and 'medical degree' in education_levels:
        high_school = df[df['education'] == 'high school diploma or lower'][aff_col].dropna()
        medical = df[df['education'] == 'medical degree'][aff_col].dropna()
        
        if len(high_school) >= 2 and len(medical) >= 2:
            t_stat, p_val = ttest_ind(high_school, medical)
            bias = high_school.mean() - medical.mean()
            print(f"High School vs Medical: {bias:+.3f}, p={p_val:.4f}")
            results['edu_hs_vs_med_p'] = p_val
            
            local_tests.append({
                'test': f'Affective: {context} Education (HS vs Med)',
                'p_value': p_val,
                'effect': bias,
                'test_type': 'independent_t'
            })
            
            if p_val < 0.05:
                direction = "Higher" if bias > 0 else "Lower"
                print(f"    ✓ SIGNIFICANT (uncorrected): High school gets {direction.lower()} empathy!")
    
    # 5. DIAGNOSIS ANALYSIS
    print(f"\n5. Medical Diagnosis Analysis")
    dx_stats = df.groupby('diagnosis')[aff_col].agg(['count', 'mean', 'std']).round(3)
    print("Diagnosis statistics (Affective Empathy):")
    print(dx_stats.head(10))  # Show top 10
    
    # Test diagnosis differences
    diagnoses = df['diagnosis'].unique()
    dx_data = []
    dx_means = []
    dx_names = []
    
    for dx in diagnoses:
        scores = df[df['diagnosis'] == dx][aff_col].dropna()
        if len(scores) >= 2:
            dx_data.append(scores)
            dx_means.append(scores.mean())
            dx_names.append(dx)
            
    if len(dx_data) >= 2:
        f_stat, p_val = f_oneway(*dx_data)
        print(f"Diagnosis ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
        results['diagnosis_anova_p'] = p_val
        
        local_tests.append({
            'test': f'Affective: {context} Diagnosis',
            'p_value': p_val,
            'effect': f_stat,
            'test_type': 'anova'
        })
        
        if p_val < 0.05:
            print(f"    ✓ SIGNIFICANT diagnosis differences (uncorrected)!")
            max_idx = np.argmax(dx_means)
            min_idx = np.argmin(dx_means)
            print(f"    Highest empathy: {dx_names[max_idx]} ({dx_means[max_idx]:.3f})")
            print(f"    Lowest empathy: {dx_names[min_idx]} ({dx_means[min_idx]:.3f})")
    
    return results, local_tests

# Run demographic analysis for all combinations
print(f"\n🔄 Running all response-rater combinations...")

claude_gpt_results, claude_gpt_tests = analyze_demographic_patterns(claude_df, "Claude", "GPT")
all_tests.extend(claude_gpt_tests)

claude_claude_results, claude_claude_tests = analyze_demographic_patterns(claude_df, "Claude", "Claude")
all_tests.extend(claude_claude_tests)

gpt_gpt_results, gpt_gpt_tests = analyze_demographic_patterns(gpt_df, "GPT", "GPT")
all_tests.extend(gpt_gpt_tests)

gpt_claude_results, gpt_claude_tests = analyze_demographic_patterns(gpt_df, "GPT", "Claude")
all_tests.extend(gpt_claude_tests)

print("\n" + "=" * 80)
print("3. RESPONSE SOURCE COMPARISON")
print("=" * 80)

print(f"\n🤖 Do Claude and GPT responses receive different empathy ratings?")
print("-" * 60)

# Compare empathy ratings between Claude and GPT responses
# Using GPT as rater for fair comparison
claude_aff_gpt = claude_df['Affective Empathy Score (GPT)'].dropna()
gpt_aff_gpt = gpt_df['Affective Empathy Score (GPT)'].dropna()

claude_cog_gpt = claude_df['Cognitive Empathy Score (GPT)'].dropna()
gpt_cog_gpt = gpt_df['Cognitive Empathy Score (GPT)'].dropna()

print(f"Sample sizes: Claude responses={len(claude_aff_gpt)}, GPT responses={len(gpt_aff_gpt)}")

# Affective empathy comparison
if len(claude_aff_gpt) > 0 and len(gpt_aff_gpt) > 0:
    t_stat, p_val = ttest_ind(claude_aff_gpt, gpt_aff_gpt)
    bias = claude_aff_gpt.mean() - gpt_aff_gpt.mean()
    
    print(f"Affective Empathy (GPT rater):")
    print(f"  Claude responses: {claude_aff_gpt.mean():.3f}")
    print(f"  GPT responses: {gpt_aff_gpt.mean():.3f}")
    print(f"  Difference: {bias:+.3f}, p={p_val:.4f}")
    
    all_tests.append({
        'test': 'Response Source: Affective (GPT rater)',
        'p_value': p_val,
        'effect': bias,
        'test_type': 'independent_t'
    })
    
    if p_val < 0.05:
        winner = "Claude" if bias > 0 else "GPT"
        print(f"    ✓ SIGNIFICANT (uncorrected): {winner} responses receive higher affective empathy ratings!")

# Cognitive empathy comparison
if len(claude_cog_gpt) > 0 and len(gpt_cog_gpt) > 0:
    t_stat, p_val = ttest_ind(claude_cog_gpt, gpt_cog_gpt)
    bias = claude_cog_gpt.mean() - gpt_cog_gpt.mean()
    
    print(f"Cognitive Empathy (GPT rater):")
    print(f"  Claude responses: {claude_cog_gpt.mean():.3f}")
    print(f"  GPT responses: {gpt_cog_gpt.mean():.3f}")
    print(f"  Difference: {bias:+.3f}, p={p_val:.4f}")
    
    all_tests.append({
        'test': 'Response Source: Cognitive (GPT rater)',
        'p_value': p_val,
        'effect': bias,
        'test_type': 'independent_t'
    })
    
    if p_val < 0.05:
        winner = "Claude" if bias > 0 else "GPT"
        print(f"    ✓ SIGNIFICANT (uncorrected): {winner} responses receive higher cognitive empathy ratings!")

print("\n" + "=" * 80)
print("4. BENJAMINI-HOCHBERG CORRECTION")
print("=" * 80)

print(f"\n📊 Total tests conducted: {len(all_tests)}")
print(f"Applying Benjamini-Hochberg correction (FDR = 0.05)...")

# Convert to dataframe for easier handling
all_tests_df = pd.DataFrame(all_tests)

# Extract p-values
p_values = all_tests_df['p_value'].values

# Apply BH correction
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Add to dataframe
all_tests_df['p_adjusted'] = p_adjusted
all_tests_df['significant_uncorrected'] = p_values < 0.05
all_tests_df['significant_corrected'] = reject

print(f"\nSignificance Summary:")
print(f"  Uncorrected p<0.05: {sum(all_tests_df['significant_uncorrected'])} tests")
print(f"  After BH correction: {sum(all_tests_df['significant_corrected'])} tests")
print(f"  Lost significance: {sum(all_tests_df['significant_uncorrected']) - sum(all_tests_df['significant_corrected'])} tests")

print("\n" + "=" * 80)
print("5. RESULTS COMPARISON: BEFORE vs AFTER BH CORRECTION")
print("=" * 80)

# Show tests that changed significance status
lost_significance = all_tests_df[
    (all_tests_df['significant_uncorrected'] == True) & 
    (all_tests_df['significant_corrected'] == False)
]

stayed_significant = all_tests_df[all_tests_df['significant_corrected'] == True]

print(f"\n❌ LOST SIGNIFICANCE AFTER BH CORRECTION ({len(lost_significance)} tests):")
print("-" * 60)
if len(lost_significance) > 0:
    for idx, row in lost_significance.iterrows():
        print(f"  • {row['test']}")
        print(f"    Uncorrected p={row['p_value']:.4f} → Adjusted p={row['p_adjusted']:.4f}")
        print(f"    Effect size: {row['effect']:.3f}")
else:
    print("  None! All significant tests survived correction.")

print(f"\n✅ REMAINED SIGNIFICANT AFTER BH CORRECTION ({len(stayed_significant)} tests):")
print("-" * 60)
if len(stayed_significant) > 0:
    for idx, row in stayed_significant.iterrows():
        print(f"  • {row['test']}")
        print(f"    Uncorrected p={row['p_value']:.4f} → Adjusted p={row['p_adjusted']:.4f}")
        print(f"    Effect size: {row['effect']:.3f}")
else:
    print("  None.")

print("\n" + "=" * 80)
print("6. FINAL RESULTS TABLE")
print("=" * 80)

# Sort by significance and p-value
all_tests_df_sorted = all_tests_df.sort_values(['significant_corrected', 'p_value'], 
                                               ascending=[False, True])

print("\nComplete Results (sorted by significance):")
print("=" * 80)

results_table = all_tests_df_sorted[['test', 'p_value', 'p_adjusted', 
                                     'significant_uncorrected', 'significant_corrected', 
                                     'effect']].copy()
results_table.columns = ['Test', 'p (uncorr)', 'p (BH-adj)', 'Sig (uncorr)', 'Sig (BH)', 'Effect']

print(results_table.to_string(index=False))

print("\n" + "=" * 80)
print("7. SUMMARY OF KEY FINDINGS")
print("=" * 80)

print(f"\n🎯 ROBUST FINDINGS (Survived BH correction):")
print("-" * 60)

# Group by test category
if len(stayed_significant) > 0:
    # Education findings
    edu_findings = stayed_significant[stayed_significant['test'].str.contains('Education')]
    if len(edu_findings) > 0:
        print(f"\n📚 EDUCATION EFFECTS:")
        for idx, row in edu_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Gender findings
    gender_findings = stayed_significant[stayed_significant['test'].str.contains('Gender')]
    if len(gender_findings) > 0:
        print(f"\n👥 GENDER EFFECTS:")
        for idx, row in gender_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Age findings
    age_findings = stayed_significant[stayed_significant['test'].str.contains('Age')]
    if len(age_findings) > 0:
        print(f"\n📅 AGE EFFECTS:")
        for idx, row in age_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Ethnicity findings
    eth_findings = stayed_significant[stayed_significant['test'].str.contains('Ethnicity')]
    if len(eth_findings) > 0:
        print(f"\n🌍 ETHNICITY EFFECTS:")
        for idx, row in eth_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Diagnosis findings
    dx_findings = stayed_significant[stayed_significant['test'].str.contains('Diagnosis')]
    if len(dx_findings) > 0:
        print(f"\n🏥 DIAGNOSIS EFFECTS:")
        for idx, row in dx_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Response source findings
    response_findings = stayed_significant[stayed_significant['test'].str.contains('Response Source')]
    if len(response_findings) > 0:
        print(f"\n🤖 RESPONSE SOURCE EFFECTS:")
        for idx, row in response_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")
    
    # Rater agreement findings
    rater_findings = stayed_significant[stayed_significant['test'].str.contains('Rater Agreement')]
    if len(rater_findings) > 0:
        print(f"\n🤝 RATER AGREEMENT:")
        for idx, row in rater_findings.iterrows():
            print(f"  • {row['test']}: p={row['p_adjusted']:.4f}")

else:
    print("  No findings survived BH correction at α=0.05.")

print(f"\n⚠️  MARGINAL FINDINGS (Lost significance after correction):")
print("-" * 60)
if len(lost_significance) > 0:
    print(f"  {len(lost_significance)} tests were significant before but not after BH correction.")
    print(f"  These should be reported as exploratory or hypothesis-generating.")
else:
    print("  None - all significant findings were robust!")

print("\n" + "=" * 80)
print("8. WHAT CHANGED?")
print("=" * 80)

print(f"\n📈 IMPACT OF BH CORRECTION:")
print("-" * 60)

# Calculate impact metrics
original_sig = sum(all_tests_df['significant_uncorrected'])
corrected_sig = sum(all_tests_df['significant_corrected'])
survival_rate = (corrected_sig / original_sig * 100) if original_sig > 0 else 0

print(f"  • Original significant tests: {original_sig}/{len(all_tests)} ({original_sig/len(all_tests)*100:.1f}%)")
print(f"  • After BH correction: {corrected_sig}/{len(all_tests)} ({corrected_sig/len(all_tests)*100:.1f}%)")
print(f"  • Survival rate: {survival_rate:.1f}%")

if survival_rate >= 80:
    print(f"\n  ✅ HIGH SURVIVAL RATE: Most findings are robust!")
elif survival_rate >= 50:
    print(f"\n  ⚠️  MODERATE SURVIVAL: Some findings were marginal.")
else:
    print(f"\n  ❌ LOW SURVIVAL RATE: Many findings were likely false positives.")

print(f"\n💡 INTERPRETATION GUIDANCE:")
print("-" * 60)
print(f"  • REPORT as significant: Tests with p(BH-adj) < 0.05")
print(f"  • MENTION as exploratory: Tests with 0.05 < p(BH-adj) < 0.10")
print(f"  • DO NOT claim significance: Tests with p(BH-adj) ≥ 0.10")

# Save results
print("\n💾 Saving results...")
all_tests_df_sorted.to_csv('data/results/empathy_scores/empathy_results_with_BH_correction.csv', index=False)
combined_df.to_csv('data/results/empathy_scores/combined_empathy_data.csv', index=False)

print(f"  ✓ data/results/empathy_scores/empathy_results_with_BH_correction.csv")
print(f"  ✓ data/results/empathy_scores/combined_empathy_data.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE WITH BH CORRECTION")
print("=" * 80)
print(f"\n✅ You now have properly corrected results!")
print(f"✅ Camera-ready paper should report BH-corrected p-values")
print(f"✅ Add one sentence to Methods: 'Benjamini-Hochberg correction applied (FDR=0.05)'")