# -*- coding: utf-8 -*-
"""
fairness_evaluation_utils.py

Purpose:
    Provides a centralized set of functions for calculating group performance
    and fairness disparity metrics.
    UPGRADED: Now handles both multiclass and binary evaluation contexts.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_group_metrics(df: pd.DataFrame, group_cols: list, positive_class: str = 'Amateur') -> pd.DataFrame:
    """
    Calculates performance metrics for each specified group.
    Handles both binary and multiclass evaluation based on the predicted values.
    """
    results = []
    
    is_binary_prediction = df['predicted_category'].nunique() <= 2
    if is_binary_prediction:
        print("  - Performing BINARY classification evaluation.")
    else:
        print("  - Performing MULTICLASS classification evaluation.")

    df_sets = {'Overall': df}
    for col in group_cols:
        df_sets[col] = df[df[col] == 1]

    for name, subset_df in df_sets.items():
        if len(subset_df) == 0: continue
            
        if is_binary_prediction:
            # For binary, accuracy is meaningful on the binarized task
            y_true_binary = (subset_df['primary_category'] == positive_class)
            y_pred_binary = (subset_df['predicted_category'] == positive_class)
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
        else:
            # For multiclass, accuracy is on the original labels
            accuracy = accuracy_score(subset_df['primary_category'], subset_df['predicted_category'])

        # Fairness metrics like Precision/Recall are always based on the binary case
        y_true_fairness = (subset_df['primary_category'] == positive_class)
        y_pred_fairness = (subset_df['predicted_category'] == positive_class)
        
        metrics = {
            'Group': name,
            'Accuracy': accuracy,
            'Precision': precision_score(y_true_fairness, y_pred_fairness, zero_division=0),
            'Recall (TPR)': recall_score(y_true_fairness, y_pred_fairness, zero_division=0),
            'F1-Score': f1_score(y_true_fairness, y_pred_fairness, zero_division=0),
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