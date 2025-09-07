# -*- coding: utf-8 -*-
"""
06_stratified_splitting.py

Purpose:
    Splits the master corpus into training, validation, and test sets. This is
    the foundational step for the entire modeling phase.

Core Logic:
    1.  Loads the master ml_corpus.parquet file.
    2.  Creates a single, robust 'stratification_group' column by assigning each
        video to a primary intersectional group. This avoids errors caused by
        extremely rare class combinations.
    3.  Uses this new column to perform a stable stratified split, ensuring
        all key groups are proportionally represented.
    4.  Saves the video_ids for each split into separate CSV files.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

CONFIG = load_config()

# Define paths
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
OUTPUT_DIR = Path(CONFIG['paths']['data'])

# Define split ratios
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25 # 25% of the remaining 80% -> 20% of the total

# --- 2. Main Execution ---

def main():
    """Main function to perform the stratified split."""
    print("--- Starting Step 06: Stratified Data Splitting ---")

    print(f"Loading corpus from {CORPUS_PATH}...")
    df = pd.read_parquet(CORPUS_PATH)

    # Create a single, stable column for stratification by prioritizing key
    # intersections and grouping all others into an 'Other' category.
    print("Creating a robust stratification key...")
    
    conditions = [
        (df['intersectional_black_female'] == 1),
        (df['race_ethnicity_white'] == 1) & (df['gender_female'] == 1),
        (df['race_ethnicity_asian'] == 1) & (df['gender_female'] == 1),
        (df['race_ethnicity_latina'] == 1) & (df['gender_female'] == 1),
    ]
    
    choices = [
        'Black_Female',
        'White_Female',
        'Asian_Female',
        'Latina_Female',
    ]
    
    df['stratify_key'] = np.select(conditions, choices, default='Other')
    
    print(f"Stratifying based on the following groups: {df['stratify_key'].unique().tolist()}")
    print("Value counts of stratification key:\n", df['stratify_key'].value_counts())

    # First Split: Separate Test Set
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=CONFIG['reproducibility']['seed'],
        stratify=df['stratify_key']
    )
    
    # Second Split: Separate Training and Validation Sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALIDATION_SIZE,
        random_state=CONFIG['reproducibility']['seed'],
        stratify=train_val_df['stratify_key']
    )

    print("\nSplit sizes:")
    print(f"  Training set:   {len(train_df):,} records ({len(train_df)/len(df):.2%})")
    print(f"  Validation set: {len(val_df):,} records ({len(val_df)/len(df):.2%})")
    print(f"  Test set:       {len(test_df):,} records ({len(test_df)/len(df):.2%})")

    # Save only the video_ids for each split
    print("\nSaving data split IDs...")
    train_path = OUTPUT_DIR / 'train_ids.csv'
    val_path = OUTPUT_DIR / 'val_ids.csv'
    test_path = OUTPUT_DIR / 'test_ids.csv'
    
    train_df[['video_id']].to_csv(train_path, index=False)
    print(f"✓ Artefact saved: {train_path.resolve()}")
    
    val_df[['video_id']].to_csv(val_path, index=False)
    print(f"✓ Artefact saved: {val_path.resolve()}")

    test_df[['video_id']].to_csv(test_path, index=False)
    print(f"✓ Artefact saved: {test_path.resolve()}")
    
    print("\n--- Step 06: Stratified Data Splitting Completed Successfully ---")

if __name__ == '__main__':
    main()