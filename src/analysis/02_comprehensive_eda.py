# -*- coding: utf-8 -*-
"""
02_comprehensive_eda.py

Purpose:
    Performs a thorough and multi-faceted Exploratory Data Analysis (EDA) on
    the master corpus. This script is designed to provide a deep, quantitative
    foundation for the dissertation by addressing specific research questions
    about representation, intersectionality, and outcome disparities (ratings).

Core Analyses:
    1.  Overall Corpus Statistics: Total video count and distribution of the
        top video categories.
    2.  Full Intersectional Representation: Systematically creates all race x gender
        intersections and quantifies their prevalence in the corpus.
    3.  Rating Disparity Analysis: Compares the distribution of video ratings
        across key female intersectional groups (Black, White, Asian, Latina)
        to identify potential allocative disparities.
"""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import itertools

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
NARRATIVE_PATH = Path(CONFIG['paths']['narratives']) / 'automated' / '02_comprehensive_eda_summary.md'
DATA_DIR = Path(CONFIG['paths']['data'])
FIGURES_DIR = Path(CONFIG['paths']['figures'])
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Analysis Functions ---

def analyze_category_distribution(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Analyzes the distribution of the most frequent video categories."""
    print("Analyzing top video category distribution...")
    df_categories = df['categories'].str.split(',').explode()
    top_categories = df_categories.value_counts().nlargest(top_n).reset_index()
    top_categories.columns = ['Category', 'Video Count']
    top_categories['Percentage'] = (top_categories['Video Count'] / len(df)) * 100
    print("✓ Category distribution analysis complete.")
    return top_categories

def analyze_full_intersections(df: pd.DataFrame) -> pd.DataFrame:
    """Systematically creates and analyzes all race x gender intersections."""
    print("Analyzing all race x gender intersections...")
    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    gender_cols = sorted([col for col in df.columns if col.startswith('gender_')])
    
    intersections = []
    for race_col, gender_col in itertools.product(race_cols, gender_cols):
        intersection_name = f"{race_col.split('_')[-1].capitalize()} x {gender_col.split('_')[-1].capitalize()}"
        count = len(df[(df[race_col] == 1) & (df[gender_col] == 1)])
        if count > 0:
            intersections.append({
                'Intersection': intersection_name,
                'Count': count,
                'Percentage': (count / len(df)) * 100
            })
            
    df_intersections = pd.DataFrame(intersections).sort_values('Count', ascending=False).reset_index(drop=True)
    print("✓ Full intersectional analysis complete.")
    return df_intersections

def analyze_rating_disparities(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyzes and compares rating distributions for key female intersections."""
    print("Analyzing rating disparities across key female intersections...")
    intersections_of_interest = {
        'Black Women': (df['race_ethnicity_black'] == 1) & (df['gender_female'] == 1),
        'White Women': (df['race_ethnicity_white'] == 1) & (df['gender_female'] == 1),
        'Asian Women': (df['race_ethnicity_asian'] == 1) & (df['gender_female'] == 1),
        'Latina Women': (df['race_ethnicity_latina'] == 1) & (df['gender_female'] == 1),
    }
    
    rating_data = [df[condition][['rating']].assign(Group=name) for name, condition in intersections_of_interest.items() if not df[condition].empty]
    df_plot = pd.concat(rating_data, ignore_index=True)
    df_stats = df_plot.groupby('Group')['rating'].describe()
    
    print("✓ Rating disparity analysis complete.")
    return df_plot, df_stats

# --- 3. Visualization Functions (with robust palette handling) ---

def get_smart_palette(base_palette: list, num_categories: int) -> list:
    """Selects from a curated list or generates a new one if more colors are needed."""
    if num_categories > len(base_palette):
        print(f"  - Info: Generating a sequential palette for {num_categories} categories.")
        return sns.color_palette("plasma", n_colors=num_categories)
    return base_palette[:num_categories]

@plot_dual_theme(section='eda')
def plot_categories(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Plots the top N video categories."""
    active_palette = get_smart_palette(palette, len(data['Category'].unique()))
    sns.barplot(x='Video Count', y='Category', hue='Category', data=data, palette=active_palette, legend=False, ax=ax)
    ax.set_title(f'Top {len(data)} Most Frequent Video Categories')
    ax.set_xlabel('Total Video Count')
    ax.set_ylabel(None)

@plot_dual_theme(section='eda')
def plot_intersections(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Plots the prevalence of different race x gender intersections."""
    active_palette = get_smart_palette(palette, len(data['Intersection'].unique()))
    sns.barplot(x='Percentage', y='Intersection', hue='Intersection', data=data, palette=active_palette, legend=False, ax=ax)
    ax.set_title('Representation of Race x Gender Intersections')
    ax.set_xlabel('Percentage of Total Videos (%)')
    ax.set_ylabel(None)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))

@plot_dual_theme(section='eda')
def plot_ratings(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Creates a box plot to compare rating distributions."""
    active_palette = get_smart_palette(palette, len(data['Group'].unique()))
    sns.boxplot(x='rating', y='Group', hue='Group', data=data, palette=active_palette, legend=False, ax=ax)
    ax.set_title('Comparison of Video Rating Distributions by Group')
    ax.set_xlabel('Video Rating')
    ax.set_ylabel(None)

# --- 4. Main Execution Pipeline ---

def main():
    print("--- Starting Step 02: Comprehensive & Enhanced EDA ---")
    
    df = pd.read_parquet(CORPUS_PATH)
    total_videos = len(df)
    
    df_top_categories = analyze_category_distribution(df)
    df_intersections = analyze_full_intersections(df)
    df_ratings_plot, df_ratings_stats = analyze_rating_disparities(df)

    print("\nSaving data artefacts...")
    df_top_categories.to_csv(DATA_DIR / 'eda_top_categories.csv', index=False)
    df_intersections.to_csv(DATA_DIR / 'eda_intersectional_representation.csv', index=False)
    df_ratings_stats.to_csv(DATA_DIR / 'eda_rating_disparities_stats.csv')
    for path in [DATA_DIR / 'eda_top_categories.csv', DATA_DIR / 'eda_intersectional_representation.csv', DATA_DIR / 'eda_rating_disparities_stats.csv']:
        print(f"✓ Artefact saved: {path.resolve()}")

    print("\nGenerating visualizations...")
    plot_categories(data=df_top_categories, save_path=str(FIGURES_DIR / 'eda_top_categories_bar'), figsize=(10, 8))
    plot_intersections(data=df_intersections, save_path=str(FIGURES_DIR / 'eda_intersections_bar'), figsize=(10, 8))
    plot_ratings(data=df_ratings_plot, save_path=str(FIGURES_DIR / 'eda_ratings_boxplot'), figsize=(10, 6))

    print("\nGenerating narrative summary and LaTeX tables...")
    summary = f"""
# Automated Summary: Comprehensive Exploratory Data Analysis

This report details the initial findings from the corpus of **{total_videos:,}** videos.

## 1. Overall Corpus Composition
The corpus contains a wide variety of content. The most frequent category is **"{df_top_categories.iloc[0]['Category']}"**, appearing in **{df_top_categories.iloc[0]['Video Count']:,}** videos ({df_top_categories.iloc[0]['Percentage']:.2f}% of the total).

## 2. Intersectional Representation (Race x Gender)
A full analysis of race and gender intersections reveals significant representational disparities.
- **Most Represented Intersection**: `{df_intersections.iloc[0]['Intersection']}` constitutes **{df_intersections.iloc[0]['Percentage']:.2f}%** of the corpus ({df_intersections.iloc[0]['Count']:,} videos).
- **Least Represented Intersection**: `{df_intersections.iloc[-1]['Intersection']}` constitutes just **{df_intersections.iloc[-1]['Percentage']:.2f}%** of the corpus ({df_intersections.iloc[-1]['Count']:,} videos).
- **Primary Focus (Black Women)**: The `Black x Female` intersection represents **{df_intersections[df_intersections['Intersection'] == 'Black x Female']['Percentage'].values[0]:.2f}%** of all videos.

## 3. Rating Disparity Analysis
A comparison of video ratings across key female intersectional groups indicates potential differences in outcomes.
- **Highest Median Rating**: The group with the highest median rating is **{df_ratings_stats['50%'].idxmax()}** (Median = **{df_ratings_stats['50%'].max():.2f}**).
- **Lowest Median Rating**: The group with the lowest median rating is **{df_ratings_stats['50%'].idxmin()}** (Median = **{df_ratings_stats['50%'].min():.2f}**).

These initial figures provide strong quantitative evidence of both representational and outcome disparities, which will be the focus of subsequent fairness analyses.
"""
    with open(NARRATIVE_PATH, 'w') as f: f.write(summary)
    print(f"✓ Artefact saved: {NARRATIVE_PATH.resolve()}")

    dataframe_to_latex_table(df_top_categories.set_index('Category'), str(TABLES_DIR / 'eda_top_categories.tex'), "Top 15 Most Frequent Video Categories.", "tab:top-categories")
    dataframe_to_latex_table(df_intersections.set_index('Intersection'), str(TABLES_DIR / 'eda_intersections.tex'), "Representation of Race x Gender Intersections.", "tab:intersections")
    dataframe_to_latex_table(df_ratings_stats, str(TABLES_DIR / 'eda_rating_stats.tex'), "Summary Statistics of Video Ratings by Group.", "tab:rating-stats")
    
    print("\n--- Step 02: Comprehensive & Enhanced EDA Completed Successfully ---")

if __name__ == '__main__':
    main()