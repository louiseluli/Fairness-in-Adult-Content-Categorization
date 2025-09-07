# -*- coding: utf-8 -*-
"""
16_temporal_and_engagement_analysis.py

Purpose:
    Conducts a deep analysis of temporal and engagement-based biases. This
    script addresses a critical gap by leveraging the 'publish_date', 'views',
    and 'ratings' columns to understand how biases evolve over time and how
    they manifest in user engagement patterns.

Core Analyses:
    1.  Temporal Representation Analysis: Tracks the prevalence of key
        intersectional groups from 2010 to 2024 to identify trends.
    2.  Temporal Harm Analysis: Measures the prevalence of derogatory language
        (from HurtLex 'cds' category) over time.
    3.  Engagement Disparity Analysis: Compares the distribution of view counts
        for different intersectional groups using a log-scaled boxenplot for
        clarity with skewed data.

This script provides critical evidence for a distinction-level analysis, adding
a dynamic and impact-oriented dimension to the dissertation.
"""

import sys
import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
LEXICA_PATH = Path(CONFIG['project']['root']) / 'config' / 'abusive_lexica'
OUTPUT_DIR = Path(CONFIG['paths']['data'])
FIGURES_DIR = Path(CONFIG['paths']['figures'])
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Analysis Functions ---

def temporal_representation_analysis(df: pd.DataFrame):
    """Tracks the prevalence of key intersectional groups over time."""
    print("Analyzing temporal representation trends...")
    
    # Create intersectional groups for analysis
    df['group_black_women'] = ((df['race_ethnicity_black'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_white_women'] = ((df['race_ethnicity_white'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_asian_women'] = ((df['race_ethnicity_asian'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_latina_women'] = ((df['race_ethnicity_latina'] == 1) & (df['gender_female'] == 1)).astype(int)
    
    groups_to_track = ['group_black_women', 'group_white_women', 'group_asian_women', 'group_latina_women']
    
    # Group by year and calculate the mean prevalence of each group
    temporal_df = df.groupby('publish_year')[groups_to_track].mean() * 100
    temporal_df.rename(columns={
        'group_black_women': 'Black Women', 'group_white_women': 'White Women',
        'group_asian_women': 'Asian Women', 'group_latina_women': 'Latina Women'
    }, inplace=True)
    
    return temporal_df

def temporal_harm_analysis(df: pd.DataFrame, lexicon: pd.DataFrame):
    """Tracks the prevalence of derogatory terms over time."""
    print("Analyzing temporal trends in harmful language...")
    
    # Isolate derogatory terms from HurtLex
    derogatory_terms = lexicon[lexicon['category'] == 'cds']['term'].tolist()
    pattern = r'\b(' + '|'.join(re.escape(term) for term in derogatory_terms) + r')\b'
    
    # Create a flag for videos containing derogatory terms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df['has_derogatory_term'] = df['model_input_text'].str.contains(pattern, regex=True, na=False)
        
    # Group by year and calculate the percentage of videos with derogatory terms
    harm_trend = df.groupby('publish_year')['has_derogatory_term'].mean() * 100
    return harm_trend.reset_index()


def engagement_disparity_analysis(df: pd.DataFrame):
    """Compares view count distributions across intersectional groups."""
    print("Analyzing engagement disparities (views)...")
    
    # Use the same intersectional groups as defined before
    df['group'] = np.select(
        [df['group_black_women'] == 1, df['group_white_women'] == 1, df['group_asian_women'] == 1, df['group_latina_women'] == 1],
        ['Black Women', 'White Women', 'Asian Women', 'Latina Women'],
        default='Other'
    )
    
    # Filter for the groups of interest and apply a log transform to views for better visualization
    df_engagement = df[df['group'] != 'Other'].copy()
    df_engagement['log_views'] = np.log1p(df_engagement['views'])
    
    return df_engagement

# --- 3. Visualization Functions ---

@plot_dual_theme(section='eda')
def plot_temporal_representation(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    data.plot(kind='line', ax=ax, color=palette, marker='o', linestyle='-')
    ax.set_title('Representation of Female Intersectional Groups Over Time')
    ax.set_xlabel('Year of Publication')
    ax.set_ylabel('Percentage of Annual Videos (%)')
    ax.legend(title='Group')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

@plot_dual_theme(section='fairness')
def plot_temporal_harm(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    sns.lineplot(x='publish_year', y='has_derogatory_term', data=data, ax=ax, color=palette[2], marker='o')
    ax.set_title('Prevalence of Derogatory Terms (HurtLex "cds") Over Time')
    ax.set_xlabel('Year of Publication')
    ax.set_ylabel('Videos with Derogatory Terms (%)')

@plot_dual_theme(section='eda')
def plot_engagement_views(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    sns.boxenplot(x='log_views', y='group', hue='group', data=data, palette=palette, legend=False, ax=ax)
    ax.set_title('Distribution of Video View Counts by Intersectional Group')
    ax.set_xlabel('Views (Log Scale)')
    ax.set_ylabel(None)

# --- 4. Main Execution ---

def main():
    """Main function to run the temporal and engagement analysis pipeline."""
    start_time = time.monotonic()
    print("--- Starting Step 16: Temporal & Engagement Bias Analysis ---")

    # Load data
    df = pd.read_parquet(CORPUS_PATH)
    hurtlex_lexicon = pd.read_csv(LEXICA_PATH / 'hurtlex_EN.tsv', sep='\t')
    
    # Prepare data
    df['publish_year'] = pd.to_datetime(df['publish_date'], errors='coerce').dt.year
    df.dropna(subset=['publish_year'], inplace=True)
    df['publish_year'] = df['publish_year'].astype(int)
    df = df[(df['publish_year'] >= 2010) & (df['publish_year'] <= 2024)]

    # Run analyses
    df_temporal_rep = temporal_representation_analysis(df)
    df_temporal_harm = temporal_harm_analysis(df, hurtlex_lexicon)
    df_engagement = engagement_disparity_analysis(df)

    # Save data artifacts
    print("\nSaving data artefacts...")
    df_temporal_rep.to_csv(OUTPUT_DIR / 'analysis_temporal_representation.csv')
    print(f"✓ Artefact saved: {(OUTPUT_DIR / 'analysis_temporal_representation.csv').resolve()}")
    df_temporal_harm.to_csv(OUTPUT_DIR / 'analysis_temporal_harm.csv', index=False)
    print(f"✓ Artefact saved: {(OUTPUT_DIR / 'analysis_temporal_harm.csv').resolve()}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_temporal_representation(data=df_temporal_rep, save_path=str(FIGURES_DIR / 'analysis_temporal_representation'))
    plot_temporal_harm(data=df_temporal_harm, save_path=str(FIGURES_DIR / 'analysis_temporal_harm'))
    plot_engagement_views(data=df_engagement, save_path=str(FIGURES_DIR / 'analysis_engagement_views'))
    
    # Generate LaTeX Tables
    dataframe_to_latex_table(df_temporal_rep, str(TABLES_DIR / 'analysis_temporal_representation.tex'), "Annual Representation (%) of Female Intersectional Groups.", "tab:temp-rep")
    
    end_time = time.monotonic()
    print(f"\n--- Step 16: Completed in {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    main()