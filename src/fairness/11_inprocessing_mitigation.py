# -*- coding: utf-8 -*-
"""
11_inprocessing_mitigation.py

Purpose:
    Implements and evaluates an in-processing bias mitigation technique using
    the Exponentiated Gradient method from the Fairlearn library.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table
from src.fairness.fairness_evaluation_utils import calculate_group_metrics, calculate_fairness_disparities

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
TRAIN_IDS_PATH = DATA_DIR / 'train_ids.csv'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'

OUTPUT_DIR = Path(CONFIG['paths']['outputs'])
# MODEL_PATH = OUTPUT_DIR / 'models' / 'rf_inprocessing.joblib' # Disabled due to large file size
PREDICTIONS_PATH = OUTPUT_DIR / 'data' / 'rf_inprocessing_val_predictions.csv'
METRICS_PATH = OUTPUT_DIR / 'data' / 'fairness_group_metrics_inprocessing_rf.csv'
DISPARITIES_PATH = OUTPUT_DIR / 'data' / 'fairness_disparities_inprocessing_rf.csv'
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
    print("--- Starting Step 11: In-processing Mitigation (Exponentiated Gradient) ---")

    df = pd.read_parquet(CORPUS_PATH)
    train_ids = pd.read_csv(TRAIN_IDS_PATH)['video_id']
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    df = prepare_data(df)
    
    train_df = df[df['video_id'].isin(train_ids)]
    val_df = df[df['video_id'].isin(val_ids)].copy()

    X_train, y_train = train_df, train_df['target']
    X_val, y_val = val_df, val_df['target']

    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    sensitive_features_train = train_df[race_cols].idxmax(axis=1)

    text_features = 'model_input_text'
    numeric_features = ['duration', 'views', 'rating', 'ratings']
    
    preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(stop_words='english', max_features=2000), text_features),
        ('numeric', StandardScaler(), numeric_features)
    ])
    
    print("Preprocessing data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    print("✓ Data preprocessing complete.")

    # --- FIX 1: Convert sparse matrix to dense array for Fairlearn ---
    print("Converting data to dense format for Fairlearn...")
    X_train_dense = X_train_transformed.toarray()
    X_val_dense = X_val_transformed.toarray()
    print("✓ Data conversion complete.")

    print("Training in-processing (Exponentiated Gradient) model...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=CONFIG['reproducibility']['seed'], n_jobs=-1)
    constraint = DemographicParity()
    
    mitigator = ExponentiatedGradient(estimator=estimator, constraints=constraint)
    mitigator.fit(X_train_dense, y_train, sensitive_features=sensitive_features_train)
    print("✓ Model training complete.")

    print("Evaluating mitigated model on validation set...")
    y_pred = mitigator.predict(X_val_dense)
    
    val_df['predicted_category'] = np.where(y_pred == 1, 'Amateur', 'Not Amateur')

    df_group_metrics = calculate_group_metrics(val_df, race_cols)
    print("\nMitigated Model - Group-wise Performance:\n", df_group_metrics)
    
    privileged_group = 'race_ethnicity_white'
    df_disparities = calculate_fairness_disparities(df_group_metrics, privileged_group)
    print("\nMitigated Model - Fairness Disparities:\n", df_disparities)
    
    print("\nSaving artefacts...")
    # --- FIX 2: Disable saving the large model file to prevent disk space error ---
    # MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # joblib.dump(mitigator, MODEL_PATH)
    print(f"  - Model saving disabled for this script due to large file size.")
    
    val_df[['video_id', 'primary_category', 'predicted_category']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_PATH.resolve()}")
    
    df_group_metrics.to_csv(METRICS_PATH)
    print(f"✓ Metrics saved: {METRICS_PATH.resolve()}")
    
    df_disparities.to_csv(DISPARITIES_PATH)
    print(f"✓ Disparities saved: {DISPARITIES_PATH.resolve()}")

    dataframe_to_latex_table(
        df=df_disparities,
        save_path=str(TABLES_DIR / 'fairness_disparities_inprocessing_rf.tex'),
        caption="Fairness Disparity Metrics for In-Processing (EG) RF Model.",
        label="tab:fairness-disparities-inprocessing"
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Step 11: In-processing Mitigation Completed in {duration/60:.2f} minutes ---")

if __name__ == '__main__':
    main()