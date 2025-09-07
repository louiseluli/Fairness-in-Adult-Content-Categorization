# -*- coding: utf-8 -*-
"""
08_comprehensive_evaluation.py

Purpose:
    Performs a comprehensive fairness evaluation of a baseline model's
    predictions. It calculates a suite of standard group fairness metrics to
    quantify performance disparities across different racial groups.

Core Analyses:
    1.  Loads the model's predictions and the original corpus data.
    2.  Merges predictions with protected group information.
    3.  Calculates key performance metrics (accuracy, precision, recall, F1)
        for each racial group individually.
    4.  Calculates standard fairness metrics based on these group-wise
        performance scores, including:
        - Accuracy Disparity
        - Equal Opportunity Difference (based on True Positive Rate)
        - Predictive Parity Difference (based on Precision)

This script provides the core evidence for quantifying model bias as required
by RQ2 and RQ3.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
PREDICTIONS_PATH = DATA_DIR / 'rf_baseline_val_predictions.csv'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'

OUTPUT_DIR = Path(CONFIG['paths']['outputs']) / 'data'
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Fairness Metric Calculation ---

def calculate_group_metrics(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Calculates performance metrics for each specified group."""
    print("Calculating performance metrics for each protected group...")
    
    results = []
    # Ensure target columns are boolean for metric calculations
    y_true = df['is_correct']
    
    # Calculate overall metrics
    overall_metrics = {
        'Group': 'Overall',
        'Accuracy': accuracy_score(y_true, [True]*len(y_true)), # Accuracy is implicit
        'Precision': precision_score(y_true, df['y_pred'], average='macro', zero_division=0),
        'Recall (TPR)': recall_score(y_true, df['y_pred'], average='macro', zero_division=0),
        'F1-Score': f1_score(y_true, df['y_pred'], average='macro', zero_division=0),
        'Count': len(df)
    }
    results.append(overall_metrics)
    
    # Calculate metrics for each group
    for col in group_cols:
        group_df = df[df[col] == 1]
        if len(group_df) == 0:
            continue
        
        group_y_true = group_df['is_correct']
        group_y_pred = group_df['y_pred']
        
        metrics = {
            'Group': col,
            'Accuracy': accuracy_score(group_y_true, [True]*len(group_y_true)),
            'Precision': precision_score(group_y_true, group_y_pred, average='macro', zero_division=0),
            'Recall (TPR)': recall_score(group_y_true, group_y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(group_y_true, group_y_pred, average='macro', zero_division=0),
            'Count': len(group_df)
        }
        results.append(metrics)
        
    df_results = pd.DataFrame(results).set_index('Group').round(3)
    print("✓ Group metrics calculation complete.")
    return df_results

def calculate_fairness_disparities(df_metrics: pd.DataFrame, privileged_group: str) -> pd.DataFrame:
    """Calculates fairness disparities relative to a privileged group."""
    print(f"Calculating fairness disparities relative to '{privileged_group}'...")
    
    privileged_metrics = df_metrics.loc[privileged_group]
    disparities = []
    
    for group, metrics in df_metrics.iterrows():
        if group == privileged_group or group == 'Overall':
            continue
            
        disparity_data = {
            'Comparison Group': group,
            'Accuracy Disparity': privileged_metrics['Accuracy'] - metrics['Accuracy'],
            'Equal Opportunity Difference': privileged_metrics['Recall (TPR)'] - metrics['Recall (TPR)'],
            'Predictive Parity Difference': privileged_metrics['Precision'] - metrics['Precision']
        }
        disparities.append(disparity_data)
        
    df_disparities = pd.DataFrame(disparities).set_index('Comparison Group').round(3)
    print("✓ Fairness disparity calculation complete.")
    return df_disparities

# --- 3. Main Execution ---

def main():
    """Main function to run the fairness evaluation pipeline."""
    print("--- Starting Step 08: Comprehensive Fairness Evaluation ---")
    
    # Load data
    df_corpus = pd.read_parquet(CORPUS_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    # Filter for validation set and merge predictions
    df_val = df_corpus[df_corpus['video_id'].isin(val_ids)].merge(df_preds, on='video_id')
    
    # Prepare data for metric calculation
    # We need to binarize the multiclass problem for standard fairness metrics
    # A common approach is to pick one positive class, e.g., 'Amateur'
    positive_class = 'Amateur'
    df_val['y_true'] = (df_val['primary_category'] == positive_class)
    df_val['y_pred'] = (df_val['predicted_category'] == positive_class)
    df_val['is_correct'] = (df_val['primary_category'] == df_val['predicted_category'])
    
    race_cols = sorted([col for col in df_val.columns if col.startswith('race_ethnicity_')])
    
    # Calculate group-wise performance
    df_group_metrics = calculate_group_metrics(df_val, race_cols)
    print("\nGroup-wise Performance Metrics:\n", df_group_metrics)
    
    # Calculate fairness disparities
    # We define the majority group as the privileged group for comparison
    privileged_group = 'race_ethnicity_white'
    df_disparities = calculate_fairness_disparities(df_group_metrics, privileged_group)
    print("\nFairness Disparity Metrics:\n", df_disparities)
    
    # Save artifacts
    print("\nSaving artefacts...")
    group_metrics_path = OUTPUT_DIR / 'fairness_group_metrics_rf.csv'
    disparities_path = OUTPUT_DIR / 'fairness_disparities_rf.csv'

    df_group_metrics.to_csv(group_metrics_path)
    print(f"✓ Artefact saved: {group_metrics_path.resolve()}")
    df_disparities.to_csv(disparities_path)
    print(f"✓ Artefact saved: {disparities_path.resolve()}")
    
    # Generate LaTeX tables
    dataframe_to_latex_table(
        df=df_group_metrics,
        save_path=str(TABLES_DIR / 'fairness_group_metrics_rf.tex'),
        caption="Group-wise Performance Metrics for Baseline RF Model.",
        label="tab:fairness-group-metrics"
    )
    dataframe_to_latex_table(
        df=df_disparities,
        save_path=str(TABLES_DIR / 'fairness_disparities_rf.tex'),
        caption=f"Fairness Disparity Metrics for Baseline RF Model (relative to '{privileged_group}').",
        label="tab:fairness-disparities",
        note="Positive values indicate the privileged group performed better. Equal Opportunity Difference is based on True Positive Rate (Recall)."
    )
    
    print("\n--- Step 08: Comprehensive Fairness Evaluation Completed Successfully ---")

if __name__ == '__main__':
    main()