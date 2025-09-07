# -*- coding: utf-8 -*-
"""
05_statistical_bias_tests.py

Purpose:
    Performs rigorous statistical tests to determine if the disparities
    observed during the EDA are statistically significant. This script moves
    the analysis from descriptive to inferential, providing the statistical
    evidence needed to make robust claims about bias.

Core Analyses:
    1.  Representational Bias Test (Chi-squared): Tests whether the observed
        distribution of videos across racial groups is significantly different
        from a hypothetical fair (uniform) distribution.
    2.  Rating Disparity Test (Mann-Whitney U): Tests whether the observed
        differences in rating distributions between key intersectional groups
        are statistically significant.
    3.  Effect Size Calculation: For significant results, calculates an
        appropriate effect size (e.g., Cohen's d) to measure the magnitude
        of the disparity.

This script provides the core statistical evidence for RQ3 and demonstrates a
high level of academic and methodological rigor.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
import itertools

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
OUTPUT_DIR = Path(CONFIG['paths']['data'])
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Statistical Analysis Functions ---

def test_representation_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a Chi-squared test for goodness-of-fit on racial representation.
    Tests the null hypothesis that all racial groups are represented equally.
    """
    print("Performing Chi-squared test for representation bias...")
    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    
    # Observed counts: number of videos for each race
    observed_counts = df[race_cols].sum()
    
    # Expected counts under null hypothesis (uniform distribution)
    total_videos_with_race_tag = observed_counts.sum()
    expected_counts = pd.Series([total_videos_with_race_tag / len(race_cols)] * len(race_cols), index=race_cols)
    
    # Perform the test
    chi2, p_value, _, _ = chi2_contingency([observed_counts, expected_counts])
    
    result = {
        'Test': 'Chi-squared Goodness of Fit (Race Representation)',
        'Chi-squared Statistic': chi2,
        'P-value': p_value,
        'Is Significant (p < 0.05)': p_value < 0.05
    }
    print("✓ Chi-squared test complete.")
    return pd.DataFrame([result])

def cohen_d(x, y):
    """Calculates Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def test_rating_disparities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs pairwise Mann-Whitney U tests for rating disparities between key
    female intersectional groups and calculates Cohen's d as the effect size.
    """
    print("Performing pairwise Mann-Whitney U tests for rating disparities...")
    
    intersections = {
        'Black Women': df[(df['race_ethnicity_black'] == 1) & (df['gender_female'] == 1)]['rating'].dropna(),
        'White Women': df[(df['race_ethnicity_white'] == 1) & (df['gender_female'] == 1)]['rating'].dropna(),
        'Asian Women': df[(df['race_ethnicity_asian'] == 1) & (df['gender_female'] == 1)]['rating'].dropna(),
        'Latina Women': df[(df['race_ethnicity_latina'] == 1) & (df['gender_female'] == 1)]['rating'].dropna(),
    }
    
    results = []
    # Create all unique pairs of groups to compare
    for (group1_name, group1_data), (group2_name, group2_data) in itertools.combinations(intersections.items(), 2):
        if len(group1_data) < 20 or len(group2_data) < 20: # Ensure sufficient sample size
            continue
            
        # Perform the Mann-Whitney U test (non-parametric alternative to t-test)
        stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        # Calculate effect size
        effect_size = cohen_d(group1_data, group2_data)
        
        results.append({
            'Group 1': group1_name,
            'Group 2': group2_name,
            'U-statistic': stat,
            'P-value': p_value,
            'Is Significant (p < 0.05)': p_value < 0.05,
            "Cohen's d": effect_size
        })
        
    df_results = pd.DataFrame(results)
    print("✓ Pairwise rating disparity tests complete.")
    return df_results

# --- 3. Main Execution Pipeline ---

def main():
    """Main function to run the statistical testing pipeline."""
    print("--- Starting Step 05: Statistical Bias Testing ---")
    
    df = pd.read_parquet(CORPUS_PATH)
    
    # Run the statistical tests
    df_rep_test = test_representation_bias(df)
    df_rating_tests = test_rating_disparities(df)
    
    # Save artifacts
    print("\nSaving data artefacts...")
    rep_test_path = OUTPUT_DIR / 'stat_test_representation_bias.csv'
    rating_tests_path = OUTPUT_DIR / 'stat_test_rating_disparities.csv'
    
    df_rep_test.to_csv(rep_test_path, index=False)
    print(f"✓ Artefact saved: {rep_test_path.resolve()}")
    df_rating_tests.to_csv(rating_tests_path, index=False)
    print(f"✓ Artefact saved: {rating_tests_path.resolve()}")
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    dataframe_to_latex_table(
        df=df_rep_test.set_index('Test'),
        save_path=str(TABLES_DIR / 'stat_test_representation.tex'),
        caption="Chi-squared Test for Racial Representation Bias.",
        label="tab:stat-test-representation"
    )
    dataframe_to_latex_table(
        df=df_rating_tests.set_index(['Group 1', 'Group 2']),
        save_path=str(TABLES_DIR / 'stat_test_ratings.tex'),
        caption="Pairwise Mann-Whitney U Tests for Rating Disparities.",
        label="tab:stat-test-ratings",
        note="Cohen's d measures the effect size. A small effect is ~0.2, medium ~0.5, large ~0.8."
    )
    
    print("\n--- Step 05: Statistical Bias Testing Completed Successfully ---")

if __name__ == '__main__':
    main()