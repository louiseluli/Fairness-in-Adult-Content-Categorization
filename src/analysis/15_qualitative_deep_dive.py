# -*- coding: utf-8 -*-
"""
15_qualitative_deep_dive.py

Purpose:
    Performs a qualitative error analysis by sampling specific examples where
    the baseline model failed, with a special focus on the primary intersectional
    group (Black women). This provides concrete case studies for a deeper,
    more nuanced discussion of bias in the final dissertation.
"""

import sys
import pandas as pd
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
PREDICTIONS_PATH = DATA_DIR / 'rf_baseline_val_predictions.csv'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'

OUTPUT_PATH = DATA_DIR / 'qualitative_error_samples.csv'

# Define analysis parameters
NUM_SAMPLES = 50
FOCUS_GROUP = CONFIG['project_specifics']['intersection']['output_col_name']

# --- 2. Main Execution ---

def main():
    """Main function to perform the qualitative error sampling."""
    print("--- Starting Step 15: Qualitative Deep Dive ---")

    print("Loading corpus and baseline predictions...")
    df_corpus = pd.read_parquet(CORPUS_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']

    # --- FIX: Handle column name collision before merging ---
    # The 'primary_category' in the predictions file is the ground truth for the model task.
    # Let's rename it to avoid confusion with the raw 'categories' column.
    df_preds.rename(columns={'primary_category': 'true_model_category'}, inplace=True)

    # Merge predictions with the full validation set data
    df_val = df_corpus[df_corpus['video_id'].isin(val_ids)]
    df_merged = df_val.merge(df_preds, on='video_id')

    # Identify errors by comparing the model's ground truth to its prediction
    df_errors = df_merged[df_merged['true_model_category'] != df_merged['predicted_category']].copy()
    print(f"Found {len(df_errors):,} total prediction errors in the validation set.")

    # Isolate errors for the primary focus group
    df_focus_group_errors = df_errors[df_errors[FOCUS_GROUP] == 1]
    
    if len(df_focus_group_errors) == 0:
        print(f"No errors found for the focus group '{FOCUS_GROUP}'. Cannot generate samples.")
        return
        
    print(f"Found {len(df_focus_group_errors):,} errors specifically for the '{FOCUS_GROUP}' group.")

    num_to_sample = min(NUM_SAMPLES, len(df_focus_group_errors))
    print(f"Sampling {num_to_sample} of these errors for qualitative review...")
    df_samples = df_focus_group_errors.sample(n=num_to_sample, random_state=CONFIG['reproducibility']['seed'])
    
    # Select relevant columns for the output file
    output_cols = [
        'video_id', 'title', 'tags', 'true_model_category', 'predicted_category', FOCUS_GROUP
    ]
    output_cols.extend(sorted([col for col in df_samples.columns if 'race_' in col or 'gender_' in col and col != FOCUS_GROUP]))
    
    # Save the samples
    df_samples[output_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Artefact saved: {OUTPUT_PATH.resolve()}")
    print("This file contains specific examples of model failures for your qualitative analysis.")

    print("\n--- Step 15: Qualitative Deep Dive Completed Successfully ---")


if __name__ == '__main__':
    main()