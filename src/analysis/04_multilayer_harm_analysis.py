# -*- coding: utf-8 -*-
"""
04_multilayer_harm_analysis.py

Purpose:
    Performs a deep, multi-layered analysis of harmful language by leveraging
    specialized academic lexica. This script moves beyond simple term matching
    to categorize and quantify specific types of representational harm.

Core Analyses:
    1.  Lexicon Loading: Loads and normalizes HurtLex.
    2.  HurtLex Category Profiling: For each protected group, calculates the
        prevalence of the 17 specific harm categories defined in HurtLex,
        creating a detailed "harm profile".

This script directly operationalizes the measurement of "Denigration" from the
harm taxonomy and provides critical evidence for RQ1, RQ3, and RQ5.
"""

import os
import re
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
LEXICA_PATH = Path(CONFIG['project']['root']) / 'config' / 'abusive_lexica'
OUTPUT_DATA_PATH = Path(CONFIG['paths']['data']) / 'harm_category_by_group.csv'
OUTPUT_NARRATIVE_PATH = Path(CONFIG['paths']['narratives']) / 'automated' / '04_harm_analysis_summary.md'
OUTPUT_LATEX_TABLE_PATH = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables' / 'harm_category_by_group.tex'
FIG_HEATMAP_PATH = Path(CONFIG['paths']['figures']) / 'harm_category_heatmap'


# --- 2. Lexicon Loading and Processing ---

def load_lexica(path: Path) -> pd.DataFrame:
    """Loads and normalizes the HurtLex lexicon."""
    print("Loading and normalizing abusive language lexica...")
    hurtlex_path = path / 'hurtlex_EN.tsv'
    try:
        df_hurtlex = pd.read_csv(hurtlex_path, sep='\t')
        df_hurtlex = df_hurtlex[['lemma', 'category']].rename(columns={'lemma': 'term'})
        df_hurtlex['term'] = df_hurtlex['term'].str.lower().str.strip()
        print(f"✓ Loaded {len(df_hurtlex):,} terms from HurtLex.")
    except FileNotFoundError:
        print(f"✗ ERROR: HurtLex file not found at {hurtlex_path}. Halting.")
        return pd.DataFrame()
    return df_hurtlex.drop_duplicates(subset='term').reset_index(drop=True)

# --- 3. Harm Analysis ---

def create_harm_profiles(df: pd.DataFrame, df_lexicon: pd.DataFrame) -> pd.DataFrame:
    """Creates a detailed harm profile for each protected group."""
    print("Creating harm profiles for each protected group...")
    group_cols = sorted([col for col in df.columns if 'race_' in col or 'gender_' in col or 'intersectional_' in col])
    
    category_to_pattern = {}
    for category, terms in df_lexicon.groupby('category')['term']:
        pattern = r'\b(' + '|'.join(re.escape(term) for term in terms if isinstance(term, str)) + r')\b'
        category_to_pattern[category] = re.compile(pattern)
        
    harm_profiles = []
    
    for group in group_cols:
        group_df = df[df[group] == 1]
        num_group_docs = len(group_df)
        if num_group_docs == 0: continue
        
        text_series = group_df['combined_text_clean']
        
        for category, pattern in category_to_pattern.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                doc_count = text_series.str.contains(pattern, regex=True, na=False).sum()
            
            prevalence = (doc_count / num_group_docs) * 100
            harm_profiles.append({
                'Group': group, 'Harm Category': category, 'Prevalence (%)': prevalence
            })
            
    df_profiles = pd.DataFrame(harm_profiles)
    df_heatmap = df_profiles.pivot(index='Group', columns='Harm Category', values='Prevalence (%)').fillna(0)
    
    print("✓ Harm profile creation complete.")
    return df_heatmap

# --- 4. Visualization ---

@plot_dual_theme(section='fairness')
def plot_harm_heatmap(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Generates a heatmap of harm category prevalence by group."""
    sns.heatmap(
        data, ax=ax, cmap="plasma", linewidths=.5,
        annot=True, fmt=".1f", annot_kws={"size": 7}
    )
    ax.set_title('Prevalence (%) of Harm Categories by Protected Group')
    ax.set_xlabel('HurtLex Harm Category')
    ax.set_ylabel(None)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

# --- 5. Main Execution ---

def main():
    """Main function to run the harm analysis pipeline."""
    print("--- Starting Step 04: Multi-Layered Harm Analysis ---")

    df_lexicon = load_lexica(LEXICA_PATH)
    if df_lexicon.empty: return
        
    df_corpus = pd.read_parquet(CORPUS_PATH)
    df_harm_profiles = create_harm_profiles(df_corpus, df_lexicon)
    
    print("Saving data artefacts...")
    df_harm_profiles.to_csv(OUTPUT_DATA_PATH)
    print(f"✓ Artefact saved: {OUTPUT_DATA_PATH.resolve()}")
    
    print("Generating visualizations...")
    plot_harm_heatmap(data=df_harm_profiles, save_path=str(FIG_HEATMAP_PATH), figsize=(12, 10))

    print("Generating narrative summary...")
    max_val = df_harm_profiles.max().max()
    max_cat = df_harm_profiles.max().idxmax()
    max_group = df_harm_profiles[max_cat].idxmax()
    
    summary = f"""
# Automated Summary: Multi-Layered Harm Analysis

This analysis quantified the prevalence of specific harm categories (from HurtLex) across different protected groups.

## Key Findings:
- The most severe instance of linguistic harm was found for the **'{max_group}'** group.
- Content associated with this group had the highest prevalence of terms from the **'{max_cat}'** category.
- In **{max_val:.2f}%** of videos featuring the '{max_group}' group, terms from this harmful category were present.

This provides a detailed, evidence-based profile of representational harms, directly linking specific groups to academically defined categories of denigration and stereotyping.
"""
    # Create the parent directory if it doesn't exist
    OUTPUT_NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_NARRATIVE_PATH, 'w') as f: f.write(summary)
    # --- FIX ---
    # Use the correct variable name that was defined at the top of the script.
    print(f"✓ Artefact saved: {OUTPUT_NARRATIVE_PATH.resolve()}")

    dataframe_to_latex_table(
        df=df_harm_profiles, save_path=str(OUTPUT_LATEX_TABLE_PATH),
        caption="Prevalence (%) of HurtLex Harm Categories Across Protected Groups.",
        label="tab:harm-profiles",
        note="Each cell represents the percentage of a group's videos containing terms from the specified harm category."
    )
    
    print("\n--- Step 04: Multi-Layered Harm Analysis Completed Successfully ---")

if __name__ == '__main__':
    main()