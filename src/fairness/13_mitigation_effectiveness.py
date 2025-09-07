# -*- coding: utf-8 -*-
"""
13_mitigation_effectiveness.py

Purpose:
    Synthesizes the results from all trained models (baseline and mitigated)
    to create a comprehensive comparison of mitigation effectiveness. This script
    is the capstone of the mitigation phase, providing the final evidence for RQ4.

Core Logic:
    1.  Loads the saved fairness disparity metrics for each of the four models:
        - Baseline Random Forest (from Step 08)
        - Reweighed RF (from Step 10)
        - In-processing RF (from Step 11)
        - Post-processing RF (from Step 12)
    2.  Loads the overall performance metrics (e.g., accuracy) for each model.
    3.  Combines these results into a single summary DataFrame that shows the
        fairness-accuracy trade-off for each technique.
    4.  Saves the final comparison table as a CSV and a dissertation-ready
        LaTeX table.
"""

import sys
import pandas as pd
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Main Execution ---

def main():
    """Main function to synthesize and compare mitigation results."""
    print("--- Starting Step 13: Mitigation Effectiveness Analysis ---")

    # Define the paths to the four sets of disparity results
    model_disparity_files = {
        'Baseline RF': DATA_DIR / 'fairness_disparities_rf.csv',
        'Reweighed RF': DATA_DIR / 'fairness_disparities_reweighed_rf.csv',
        'In-Processing (EG)': DATA_DIR / 'fairness_disparities_inprocessing_rf.csv',
        'Post-Processing (Thresh)': DATA_DIR / 'fairness_disparities_postprocessing_rf.csv'
    }

    # Define the paths to the four sets of overall metrics
    model_metrics_files = {
        'Baseline RF': DATA_DIR / 'fairness_group_metrics_rf.csv',
        'Reweighed RF': DATA_DIR / 'fairness_group_metrics_reweighed_rf.csv',
        'In-Processing (EG)': DATA_DIR / 'fairness_group_metrics_inprocessing_rf.csv',
        'Post-Processing (Thresh)': DATA_DIR / 'fairness_group_metrics_postprocessing_rf.csv'
    }
    
    all_results = []

    print("Loading and compiling results from all models...")

    for model_name, file_path in model_disparity_files.items():
        try:
            # Load disparity and overall metrics
            df_disparity = pd.read_csv(file_path)
            df_metrics = pd.read_csv(model_metrics_files[model_name])
            
            # Extract the overall accuracy for this model
            overall_accuracy = df_metrics[df_metrics['Group'] == 'Overall']['Accuracy'].values[0]
            
            # We will focus on the disparity for the primary focus group: black_ethnicity
            black_disparity = df_disparity[df_disparity['Comparison Group'] == 'race_ethnicity_black']
            
            if not black_disparity.empty:
                result = {
                    'Model': model_name,
                    'Overall Accuracy': overall_accuracy,
                    'Accuracy Disparity (vs. White)': black_disparity['Accuracy Disparity'].values[0],
                    'Equal Opportunity Diff (vs. White)': black_disparity['Equal Opportunity Difference'].values[0]
                }
                all_results.append(result)
            else:
                 print(f"  - Warning: No disparity data found for 'race_ethnicity_black' in {model_name}.")

        except FileNotFoundError:
            print(f"✗ ERROR: Could not find result file: {file_path}. Skipping this model.")
            continue
            
    df_comparison = pd.DataFrame(all_results).set_index('Model').round(3)
    
    print("\n--- Mitigation Effectiveness Comparison ---")
    print(df_comparison)

    # Save the final comparison table
    print("\nSaving final comparison artefact...")
    comparison_path = DATA_DIR / 'mitigation_effectiveness_comparison.csv'
    df_comparison.to_csv(comparison_path)
    print(f"✓ Artefact saved: {comparison_path.resolve()}")

    # Generate the final LaTeX table for the dissertation
    dataframe_to_latex_table(
        df=df_comparison,
        save_path=str(TABLES_DIR / 'mitigation_effectiveness_comparison.tex'),
        caption="Comparison of Fairness-Accuracy Trade-offs Across Mitigation Strategies.",
        label="tab:mitigation-comparison",
        note="Disparity metrics are calculated for the 'race_ethnicity_black' group relative to the 'race_ethnicity_white' group."
    )
    
    print("\n--- Step 13: Mitigation Effectiveness Analysis Completed Successfully ---")

if __name__ == '__main__':
    main()