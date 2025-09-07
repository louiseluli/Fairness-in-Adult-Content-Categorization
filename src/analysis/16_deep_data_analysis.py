# -*- coding: utf-8 -*-
"""
16_deep_data_analysis.py

Purpose:
    Conducts a deep, multi-faceted analysis of temporal, engagement, and
    categorical biases to unlock the full potential of the dataset. This script
    addresses critical gaps required for a distinction-level dissertation.

Core Analyses:
    1.  Temporal Representation: Tracks the prevalence of key intersectional
        groups over time (2010-2024).
    2.  Engagement Disparity: Analyzes the distribution of view counts and
        the relationship between views and ratings for different groups.
    3.  Category-Group Dynamics: Creates a heatmap to visualize the prevalence
        of protected groups within the top video categories.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table
import seaborn as sns
import matplotlib.pyplot as plt

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
OUTPUT_DIR = Path(CONFIG['paths']['data'])
FIGURES_DIR = Path(CONFIG['paths']['figures'])
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Analysis Functions ---

def temporal_representation_analysis(df: pd.DataFrame):
    """Tracks the prevalence of key intersectional groups over time."""
    print("Analyzing temporal representation trends...")
    df['group_black_women'] = ((df['race_ethnicity_black'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_white_women'] = ((df['race_ethnicity_white'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_asian_women'] = ((df['race_ethnicity_asian'] == 1) & (df['gender_female'] == 1)).astype(int)
    df['group_latina_women'] = ((df['race_ethnicity_latina'] == 1) & (df['gender_female'] == 1)).astype(int)
    
    groups_to_track = ['group_black_women', 'group_white_women', 'group_asian_women', 'group_latina_women']
    temporal_df = df.groupby('publish_year')[groups_to_track].mean() * 100
    temporal_df.rename(columns={
        'group_black_women': 'Black Women', 'group_white_women': 'White Women',
        'group_asian_women': 'Asian Women', 'group_latina_women': 'Latina Women'
    }, inplace=True)
    return temporal_df

def engagement_disparity_analysis(df: pd.DataFrame):
    """Analyzes view and rating patterns across groups."""
    print("Analyzing engagement disparities...")
    df['group'] = np.select(
        [df['group_black_women'] == 1, df['group_white_women'] == 1, df['group_asian_women'] == 1, df['group_latina_women'] == 1],
        ['Black Women', 'White Women', 'Asian Women', 'Latina Women'],
        default='Other'
    )
    df_engagement = df[df['group'] != 'Other'].copy()
    df_engagement['log_views'] = np.log1p(df_engagement['views'])
    return df_engagement

def category_group_dynamics_analysis(df: pd.DataFrame, top_n_cats: int = 15):
    """Analyzes the representation of groups within top categories."""
    print("Analyzing category-group dynamics...")
    top_categories = df['primary_category'].value_counts().nlargest(top_n_cats).index
    df_top_cats = df[df['primary_category'].isin(top_categories)]
    
    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    
    # Calculate the percentage of videos in each category that belong to each racial group
    category_group_matrix = df_top_cats.groupby('primary_category')[race_cols].mean() * 100
    # Clean up column names for the plot
    category_group_matrix.columns = [col.replace('race_ethnicity_', '').capitalize() for col in category_group_matrix.columns]
    
    return category_group_matrix

# --- 3. Visualization Functions ---

@plot_dual_theme(section='eda')
def plot_temporal_representation(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    data.plot(kind='line', ax=ax, color=palette, marker='o', linestyle='-')
    ax.set_title('Representation of Female Intersectional Groups Over Time')
    ax.set_xlabel('Year of Publication')
    ax.set_ylabel('Percentage of Annual Videos (%)')
    ax.legend(title='Group')

@plot_dual_theme(section='eda')
def plot_engagement_views(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    sns.boxenplot(x='log_views', y='group', hue='group', data=data, palette=palette, legend=False, ax=ax)
    ax.set_title('Distribution of Video View Counts by Intersectional Group')
    ax.set_xlabel('Views (Log Scale)')
    ax.set_ylabel(None)

@plot_dual_theme(section='fairness')
def plot_category_group_heatmap(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    sns.heatmap(data, ax=ax, cmap="plasma", annot=True, fmt=".1f", linewidths=.5)
    ax.set_title('Racial Group Representation (%) Within Top Video Categories')
    ax.set_xlabel('Racial Group')
    ax.set_ylabel('Primary Video Category')
    plt.xticks(rotation=45, ha='right')

# --- 4. Main Execution ---

def main():
    start_time = time.monotonic()
    print("--- Starting Step 16: Deep Data Analysis ---")

    df = pd.read_parquet(CORPUS_PATH)
    df['publish_year'] = pd.to_datetime(df['publish_date'], errors='coerce').dt.year
    df.dropna(subset=['publish_year'], inplace=True)
    df['publish_year'] = df['publish_year'].astype(int)
    df = df[(df['publish_year'] >= 2010) & (df['publish_year'] <= 2024)]
    df['primary_category'] = df['categories'].str.split(',').str[0].str.strip()

    # Run analyses
    df_temporal_rep = temporal_representation_analysis(df)
    df_engagement = engagement_disparity_analysis(df)
    df_cat_group = category_group_dynamics_analysis(df)

    # Save data artifacts
    print("\nSaving data artefacts...")
    df_temporal_rep.to_csv(OUTPUT_DIR / 'deep_analysis_temporal_representation.csv')
    df_cat_group.to_csv(OUTPUT_DIR / 'deep_analysis_category_group_matrix.csv')
    print(f"✓ Artefacts saved to: {OUTPUT_DIR.resolve()}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_temporal_representation(data=df_temporal_rep, save_path=str(FIGURES_DIR / 'deep_analysis_temporal_representation'))
    plot_engagement_views(data=df_engagement, save_path=str(FIGURES_DIR / 'deep_analysis_engagement_views'))
    plot_category_group_heatmap(data=df_cat_group, save_path=str(FIGURES_DIR / 'deep_analysis_category_group_heatmap'), figsize=(12, 10))
    
    # Generate LaTeX Tables
    dataframe_to_latex_table(df_temporal_rep, str(TABLES_DIR / 'deep_analysis_temporal_representation.tex'), "Annual Representation (%) of Female Intersectional Groups.", "tab:deep-temp-rep")
    dataframe_to_latex_table(df_cat_group, str(TABLES_DIR / 'deep_analysis_category_group_matrix.tex'), "Racial Group Representation (%) Within Top Video Categories.", "tab:deep-cat-group")
    
    end_time = time.monotonic()
    print(f"\n--- Step 16: Completed in {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    main()