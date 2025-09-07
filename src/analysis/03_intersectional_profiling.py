# -*- coding: utf-8 -*-
"""
03_intersectional_profiling.py

Purpose:
    Performs a deep dive into the terms most uniquely associated with the
    primary intersectional group (Black women). This script uses Pointwise
    Mutual Information (PMI) to quantify the strength of association between
    terms in video metadata and the presence of the target group.

Core Analyses:
    1.  Calculates PMI for a vocabulary of terms (including 1- and 2-word
        phrases) with respect to the 'intersectional_black_female' group.
    2.  Identifies the top N terms with the highest PMI scores, indicating
        the strongest and most unique associations.

This analysis is a foundational step in quantifying Representational Harm
(specifically Stereotyping) and directly addresses RQ1 by uncovering the
linguistic patterns that define group representations.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths from the new config structure
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
OUTPUT_DATA_PATH = Path(CONFIG['paths']['data']) / 'pmi_intersectional_black_women.csv'
OUTPUT_NARRATIVE_PATH = Path(CONFIG['paths']['narratives']) / 'automated' / '03_pmi_summary.md'
OUTPUT_LATEX_TABLE_PATH = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables' / 'pmi_intersectional_black_women.tex'
FIG_BAR_PATH = Path(CONFIG['paths']['figures']) / 'pmi_associations_bar'

# Define analysis parameters
PMI_TARGET_GROUP = CONFIG['project_specifics']['intersection']['output_col_name']
TOP_N_TERMS = 25

# --- 2. PMI Calculation ---

def calculate_pmi(df: pd.DataFrame, target_group_col: str, text_col: str) -> pd.DataFrame:
    """
    Calculates Pointwise Mutual Information (PMI) for terms in the corpus
    with respect to a specific target group. Considers both single words
    (1-grams) and two-word phrases (2-grams).
    """
    print(f"Calculating PMI for target group: '{target_group_col}'...")
    
    vectorizer = CountVectorizer(
        stop_words='english', 
        max_features=5000, 
        binary=True,
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z0-9]+\b'
    )
    term_matrix = vectorizer.fit_transform(df[text_col])
    
    num_docs = len(df)
    p_term = np.array(term_matrix.sum(axis=0) / num_docs).flatten()
    
    p_group = df[target_group_col].sum() / num_docs
    if p_group == 0:
        print(f"✗ WARNING: Target group '{target_group_col}' has zero members. Cannot calculate PMI.")
        return pd.DataFrame()

    group_indices = df[df[target_group_col] == 1].index
    p_term_group = np.array(term_matrix[group_indices].sum(axis=0) / num_docs).flatten()

    epsilon = 1e-12
    denominator = p_term * p_group
    pmi = np.log2((p_term_group + epsilon) / (denominator + epsilon))
    
    df_pmi = pd.DataFrame({
        'Term': vectorizer.get_feature_names_out(),
        'PMI': pmi,
        'P(term)': p_term,
        'P(term|group)': p_term_group / p_group if p_group > 0 else 0
    }).sort_values('PMI', ascending=False).reset_index(drop=True)
    
    print("✓ PMI calculation complete.")
    return df_pmi

# --- 3. Visualization ---

@plot_dual_theme(section='fairness')
def plot_pmi_scores(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """
    Generates a bar plot of the top N PMI scores.
    FINAL VERSION: Fixes UserWarning by using the recommended tick labeling method.
    """
    num_categories = len(data['Term'].unique())
    
    if num_categories > len(palette):
        print(f"  - Info: Generating a sequential palette for {num_categories} categories.")
        active_palette = sns.color_palette("magma_r", n_colors=num_categories)
    else:
        active_palette = palette[:num_categories]

    sns.barplot(
        x='PMI',
        y='Term',
        hue='Term',
        data=data,
        ax=ax,
        palette=active_palette,
        legend=False
    )
    ax.set_title(f'Top {len(data)} Terms Associated with Black Women (by PMI Score)')
    ax.set_xlabel('Pointwise Mutual Information (PMI)')
    ax.set_ylabel('Term')
    
    # --- FIX for UserWarning ---
    # Get current ticks and labels
    ticks = ax.get_yticks()
    labels = [label.get_text() for label in ax.get_yticklabels()]
    # Apply wrapping to the labels
    wrapped_labels = [textwrap.fill(label, 20) for label in labels]
    # Set both the ticks and the new wrapped labels
    ax.set_yticks(ticks)
    ax.set_yticklabels(wrapped_labels)


# --- 4. Main Execution ---

def main():
    """Main function to run the PMI analysis pipeline."""
    print("--- Starting Step 03: Intersectional Profiling (PMI) ---")
    
    print(f"Loading corpus from {CORPUS_PATH}...")
    try:
        df = pd.read_parquet(CORPUS_PATH)
    except FileNotFoundError:
        print(f"✗ ERROR: Corpus file not found. Please run '01_corpus_builder.py' first.")
        return

    df_pmi = calculate_pmi(df, PMI_TARGET_GROUP, 'combined_text_clean')
    
    if df_pmi.empty:
        print("✗ Halting execution as no PMI results were generated.")
        return
        
    df_top_pmi = df_pmi.head(TOP_N_TERMS)
    
    print("Saving data artefacts...")
    df_top_pmi.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"✓ Artefact saved: {OUTPUT_DATA_PATH.resolve()}")

    print("Generating visualizations...")
    plot_pmi_scores(data=df_top_pmi, save_path=str(FIG_BAR_PATH), figsize=(10, 12))

    print("Generating narrative summary...")
    top_term = df_top_pmi.iloc[0]
    summary = f"""
# Automated Summary: Intersectional Profiling via PMI

This analysis measured the strength of association between terms and the `{PMI_TARGET_GROUP}` group using Pointwise Mutual Information (PMI). A higher PMI score indicates a stronger and more unique association.

## Key Findings:
- The term most uniquely associated with Black women in the corpus is **"{top_term['Term']}"**, with a PMI score of **{top_term['PMI']:.2f}**.
- This result provides quantitative evidence of the specific linguistic markers and potential stereotypes used to categorize this intersectional group, forming a key part of the answer to RQ1.
"""
    OUTPUT_NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_NARRATIVE_PATH, 'w') as f:
        f.write(summary)
    print(f"✓ Artefact saved: {OUTPUT_NARRATIVE_PATH.resolve()}")
    
    dataframe_to_latex_table(
        df=df_top_pmi[['Term', 'PMI', 'P(term|group)']].set_index('Term'),
        save_path=str(OUTPUT_LATEX_TABLE_PATH),
        caption=f"Top {TOP_N_TERMS} Terms Associated with the '{PMI_TARGET_GROUP}' Group by PMI.",
        label="tab:pmi-black-women",
        note="P(term|group) is the conditional probability of a term appearing given the video features the group."
    )
    
    print("\n--- Step 03: Intersectional Profiling (PMI) Completed Successfully ---")

if __name__ == '__main__':
    main()