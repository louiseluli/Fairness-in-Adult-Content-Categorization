# -*- coding: utf-8 -*-
"""
12_postprocessing_mitigation.py

Purpose:
    Implements and evaluates a post-processing bias mitigation technique using
    the ThresholdOptimizer method from the Fairlearn library. This script
    addresses RQ4 by demonstrating a method to adjust model outputs to improve
    fairness without retraining the model.

Core Logic:
    1.  Loads the ORIGINAL baseline Random Forest model and the validation data.
    2.  Uses the baseline model to get prediction scores (probabilities) for
        the validation set.
    3.  Initializes Fairlearn's ThresholdOptimizer with the baseline model's
        scores and a fairness constraint (Equalized Odds).
    4.  The optimizer finds the optimal group-specific thresholds that satisfy
        the fairness constraint.
    5.  Applies these new thresholds to get mitigated predictions.
    6.  Performs a full fairness evaluation on the new predictions and saves
        all artifacts for a three-way comparison (baseline vs. reweighing vs. post-processing).
"""

import sys
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table
from src.fairness.fairness_evaluation_utils import calculate_group_metrics, calculate_fairness_disparities

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'
# We load the ORIGINAL baseline model for this task
BASELINE_MODEL_PATH = Path(CONFIG['paths']['outputs']) / 'models' / 'rf_baseline.joblib'

OUTPUT_DIR = Path(CONFIG['paths']['outputs'])
PREDICTIONS_PATH = OUTPUT_DIR / 'data' / 'rf_postprocessing_val_predictions.csv'
METRICS_PATH = OUTPUT_DIR / 'data' / 'fairness_group_metrics_postprocessing_rf.csv'
DISPARITIES_PATH = OUTPUT_DIR / 'data' / 'fairness_disparities_postprocessing_rf.csv'
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Data Preparation ---

def prepare_data(df: pd.DataFrame, positive_class: str = 'Amateur'):
    """Binarizes the target variable for the fairness task."""
    print(f"Preparing data for modeling. Binarizing target for '{positive_class}'.")
    df_out = df.copy()
    df_out['primary_category'] = df_out['categories'].str.split(',').str[0].str.strip()
    df_out['target'] = (df_out['primary_category'] == positive_class).astype(int)
    return df_out

# --- 3. Main Execution ---

def main():
    start_time = time.monotonic()
    print("--- Starting Step 12: Post-processing Mitigation (Thresholding) ---")

    df = pd.read_parquet(CORPUS_PATH)
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    df = prepare_data(df)
    val_df = df[df['video_id'].isin(val_ids)].copy()

    X_val, y_val = val_df, val_df['target']

    # Load the pre-trained high-accuracy baseline model
    print(f"Loading baseline model from {BASELINE_MODEL_PATH}...")
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    
    # Get the baseline model's prediction scores (probabilities)
    # We need the probability of the positive class (class 1)
    print("Generating prediction scores from the baseline model...")
    y_scores = baseline_model.predict_proba(X_val)[:, 1]

    # --- Sensitive Feature Definition ---
    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    sensitive_features_val = val_df[race_cols].idxmax(axis=1)

    # --- Mitigation Step: ThresholdOptimizer ---
    print("Applying post-processing (ThresholdOptimizer) with Equalized Odds...")
    postprocess_est = ThresholdOptimizer(
        estimator=baseline_model,
        constraints="equalized_odds",
        prefit=True # We use a pre-fitted model
    )
    
    postprocess_est.fit(X_val, y_val, sensitive_features=sensitive_features_val)
    print("✓ ThresholdOptimizer fitting complete.")

    # Evaluate the mitigated model
    print("Evaluating mitigated model on validation set...")
    y_pred_mitigated = postprocess_est.predict(X_val, sensitive_features=sensitive_features_val)
    
    # --- Fairness Evaluation ---
    val_df['predicted_category'] = np.where(y_pred_mitigated == 1, 'Amateur', 'Not Amateur')

    df_group_metrics = calculate_group_metrics(val_df, race_cols)
    print("\nMitigated Model - Group-wise Performance:\n", df_group_metrics)
    
    privileged_group = 'race_ethnicity_white'
    df_disparities = calculate_fairness_disparities(df_group_metrics, privileged_group)
    print("\nMitigated Model - Fairness Disparities:\n", df_disparities)
    
    # Save artifacts
    print("\nSaving artefacts...")
    val_df[['video_id', 'primary_category', 'predicted_category']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_PATH.resolve()}")
    
    df_group_metrics.to_csv(METRICS_PATH)
    print(f"✓ Metrics saved: {METRICS_PATH.resolve()}")
    
    df_disparities.to_csv(DISPARITIES_PATH)
    print(f"✓ Disparities saved: {DISPARITIES_PATH.resolve()}")

    dataframe_to_latex_table(
        df=df_disparities,
        save_path=str(TABLES_DIR / 'fairness_disparities_postprocessing_rf.tex'),
        caption="Fairness Disparity Metrics for Post-Processing (Thresholding) RF Model.",
        label="tab:fairness-disparities-postprocessing"
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Step 12: Post-processing Mitigation Completed in {duration:.2f} seconds ---")

if __name__ == '__main__':
    main()