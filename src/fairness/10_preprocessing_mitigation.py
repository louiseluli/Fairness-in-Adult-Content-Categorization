# -*- coding: utf-8 -*-
"""
10_preprocessing_mitigation.py

Purpose:
    Implements and evaluates a pre-processing bias mitigation technique: Reweighing.
    This script addresses RQ4 by demonstrating a method to reduce bias directly
    in the data before model training.

Core Logic:
    1.  Loads the master corpus and data splits.
    2.  Calculates sample weights for the training data to balance the
        representation of different racial groups.
    3.  Retrains the same Random Forest baseline model from Step 07, but this
        time, it uses the calculated sample weights.
    4.  Performs the same comprehensive fairness evaluation as in Step 08 on
        the new, mitigated model's predictions.
    5.  Saves all artifacts, allowing for a direct comparison between the
        baseline and the reweighed model.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
TRAIN_IDS_PATH = DATA_DIR / 'train_ids.csv'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'

OUTPUT_DIR = Path(CONFIG['paths']['outputs'])
MODEL_PATH = OUTPUT_DIR / 'models' / 'rf_reweighed.joblib'
PREDICTIONS_PATH = OUTPUT_DIR / 'data' / 'rf_reweighed_val_predictions.csv'
METRICS_PATH = OUTPUT_DIR / 'data' / 'fairness_group_metrics_reweighed_rf.csv'
DISPARITIES_PATH = OUTPUT_DIR / 'data' / 'fairness_disparities_reweighed_rf.csv'
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Helper Functions (Copied from previous scripts for consistency) ---

def prepare_data(df: pd.DataFrame, target_col: str = 'categories', top_n_classes: int = 10):
    """Prepares the DataFrame for modeling."""
    print("Preparing data for modeling...")
    df['primary_category'] = df[target_col].str.split(',').str[0].str.strip()
    top_classes = df['primary_category'].value_counts().nlargest(top_n_classes).index
    df_filtered = df[df['primary_category'].isin(top_classes)].copy()
    le = LabelEncoder()
    df_filtered['target'] = le.fit_transform(df_filtered['primary_category'])
    return df_filtered, le

def calculate_group_metrics(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Calculates performance metrics for each specified group."""
    results = []
    y_true_overall = (df['primary_category'] == 'Amateur')
    y_pred_overall = (df['predicted_category'] == 'Amateur')
    
    for col in ['Overall'] + group_cols:
        if col == 'Overall':
            subset_df = df
        else:
            subset_df = df[df[col] == 1]
        
        if len(subset_df) == 0: continue
            
        y_true = (subset_df['primary_category'] == 'Amateur')
        y_pred = (subset_df['predicted_category'] == 'Amateur')
        
        metrics = {
            'Group': col,
            'Accuracy': accuracy_score(subset_df['primary_category'], subset_df['predicted_category']),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall (TPR)': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'Count': len(subset_df)
        }
        results.append(metrics)
        
    return pd.DataFrame(results).set_index('Group').round(3)

def calculate_fairness_disparities(df_metrics: pd.DataFrame, privileged_group: str) -> pd.DataFrame:
    """Calculates fairness disparities relative to a privileged group."""
    privileged_metrics = df_metrics.loc[privileged_group]
    disparities = []
    for group, metrics in df_metrics.iterrows():
        if group == privileged_group or group == 'Overall': continue
        disparity_data = {
            'Comparison Group': group,
            'Accuracy Disparity': privileged_metrics['Accuracy'] - metrics['Accuracy'],
            'Equal Opportunity Difference': privileged_metrics['Recall (TPR)'] - metrics['Recall (TPR)'],
            'Predictive Parity Difference': privileged_metrics['Precision'] - metrics['Precision']
        }
        disparities.append(disparity_data)
    return pd.DataFrame(disparities).set_index('Comparison Group').round(3)

# --- 3. Main Execution ---

def main():
    start_time = time.monotonic()
    print("--- Starting Step 10: Pre-processing Mitigation (Reweighing) ---")

    df = pd.read_parquet(CORPUS_PATH)
    train_ids = pd.read_csv(TRAIN_IDS_PATH)['video_id']
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    df, label_encoder = prepare_data(df)
    
    train_df = df[df['video_id'].isin(train_ids)].copy()
    val_df = df[df['video_id'].isin(val_ids)].copy()

    X_train, y_train = train_df, train_df['target']
    X_val, y_val = val_df, val_df['target']

    # --- Mitigation Step: Calculate Sample Weights ---
    print("Calculating sample weights for the training set...")
    race_cols = sorted([col for col in df.columns if col.startswith('race_ethnicity_')])
    # Create a single column representing the primary race for each sample
    train_df['race_group'] = train_df[race_cols].idxmax(axis=1)
    sample_weights = compute_sample_weight(class_weight='balanced', y=train_df['race_group'])
    print(f"✓ Sample weights calculated. Min weight: {sample_weights.min():.4f}, Max weight: {sample_weights.max():.4f}")

    # Define the same pipeline as the baseline model
    text_features = 'combined_text_clean'
    numeric_features = ['duration', 'views', 'rating', 'ratings']
    preprocessor = ColumnTransformer(transformers=[
        ('text', TfidfVectorizer(stop_words='english', max_features=2000), text_features),
        ('numeric', StandardScaler(), numeric_features)
    ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=CONFIG['reproducibility']['seed'], n_jobs=-1))
    ])
    
    # Train the model WITH the sample weights
    print("Training reweighed Random Forest model...")
    model_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    print("✓ Model training complete.")

    # Evaluate the mitigated model
    print("Evaluating mitigated model on validation set...")
    y_pred = model_pipeline.predict(X_val)
    
    # --- Fairness Evaluation ---
    val_df['predicted_category'] = label_encoder.inverse_transform(y_pred)
    df_group_metrics = calculate_group_metrics(val_df, race_cols)
    print("\nMitigated Model - Group-wise Performance:\n", df_group_metrics)
    
    privileged_group = 'race_ethnicity_white'
    df_disparities = calculate_fairness_disparities(df_group_metrics, privileged_group)
    print("\nMitigated Model - Fairness Disparities:\n", df_disparities)
    
    # Save all artifacts
    print("\nSaving artefacts...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH.resolve()}")
    
    val_df[['video_id', 'primary_category', 'predicted_category']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_PATH.resolve()}")
    
    df_group_metrics.to_csv(METRICS_PATH)
    print(f"✓ Metrics saved: {METRICS_PATH.resolve()}")
    
    df_disparities.to_csv(DISPARITIES_PATH)
    print(f"✓ Disparities saved: {DISPARITIES_PATH.resolve()}")

    dataframe_to_latex_table(
        df=df_disparities,
        save_path=str(TABLES_DIR / 'fairness_disparities_reweighed_rf.tex'),
        caption="Fairness Disparity Metrics for Reweighed RF Model.",
        label="tab:fairness-disparities-reweighed"
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Step 10: Pre-processing Mitigation Completed in {duration:.2f} seconds ---")

if __name__ == '__main__':
    main()