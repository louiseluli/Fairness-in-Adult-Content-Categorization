# -*- coding: utf-8 -*-
"""
07_rf_baseline.py

Purpose:
    Trains, evaluates, and saves a baseline Random Forest (RF) classifier.
    This script serves as the first step in the modeling phase, establishing a
    performance and interpretability benchmark.
"""

import sys
import pandas as pd
import joblib
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import seaborn as sns

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config, plot_dual_theme
from src.utils.academic_tables import dataframe_to_latex_table

CONFIG = load_config()

# Define paths
DATA_DIR = Path(CONFIG['paths']['data'])
CORPUS_PATH = DATA_DIR / 'ml_corpus.parquet'
TRAIN_IDS_PATH = DATA_DIR / 'train_ids.csv'
VAL_IDS_PATH = DATA_DIR / 'val_ids.csv'

OUTPUT_DIR = Path(CONFIG['paths']['outputs'])
MODEL_PATH = OUTPUT_DIR / 'models' / 'rf_baseline.joblib'
PREDICTIONS_PATH = OUTPUT_DIR / 'data' / 'rf_baseline_val_predictions.csv'
METRICS_PATH = OUTPUT_DIR / 'data' / 'rf_baseline_val_metrics.csv'
FEATURES_PATH = OUTPUT_DIR / 'data' / 'rf_baseline_feature_importances.csv'
FIGURE_PATH = Path(CONFIG['paths']['figures']) / 'rf_baseline_feature_importance'
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Data Preparation ---

def prepare_data(df: pd.DataFrame, target_col: str = 'categories', top_n_classes: int = 10):
    """Prepares the DataFrame for modeling."""
    print("Preparing data for modeling...")
    df['primary_category'] = df[target_col].str.split(',').str[0].str.strip()
    
    top_classes = df['primary_category'].value_counts().nlargest(top_n_classes).index
    df_filtered = df[df['primary_category'].isin(top_classes)].copy()
    
    le = LabelEncoder()
    df_filtered['target'] = le.fit_transform(df_filtered['primary_category'])
    
    print(f"✓ Data prepared. Target variable is 'primary_category', focusing on top {top_n_classes} classes.")
    return df_filtered, le

# --- 3. Model Training and Evaluation ---

@plot_dual_theme(section='models')
def plot_feature_importance(data: pd.DataFrame, ax=None, palette=None, **kwargs):
    """Plots the top N most important features with robust palette handling."""
    top_n = kwargs.get('top_n', 20)
    data_top = data.head(top_n)
    
    num_categories = len(data_top['Feature'].unique())
    if num_categories > len(palette):
        active_palette = sns.color_palette("plasma", n_colors=num_categories)
    else:
        active_palette = palette[:num_categories]

    sns.barplot(x='Importance', y='Feature', hue='Feature', data=data_top, palette=active_palette, legend=False, ax=ax)
    ax.set_title(f'Top {top_n} Features for Random Forest Classifier')
    ax.set_xlabel('Feature Importance (Gini Impurity)')
    ax.set_ylabel(None)

def main():
    """Main function to train and evaluate the RF baseline."""
    start_time = time.monotonic()
    print("--- Starting Step 07: Random Forest Baseline ---")

    df = pd.read_parquet(CORPUS_PATH)
    train_ids = pd.read_csv(TRAIN_IDS_PATH)['video_id']
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    df, label_encoder = prepare_data(df)
    
    train_df = df[df['video_id'].isin(train_ids)]
    val_df = df[df['video_id'].isin(val_ids)]

    X_train, y_train = train_df, train_df['target']
    X_val, y_val = val_df, val_df['target']

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
    
    print("Training Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    print("✓ Model training complete.")

    print("Evaluating model on validation set...")
    y_pred = model_pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    print("Validation Performance (rounded):\n", df_report.round(3))
    
    print("\nSaving artefacts...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH.resolve()}")
    
    val_df_with_preds = X_val.copy()
    val_df_with_preds['predicted_category'] = label_encoder.inverse_transform(y_pred)
    val_df_with_preds[['video_id', 'primary_category', 'predicted_category']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_PATH.resolve()}")
    
    df_report.to_csv(METRICS_PATH)
    print(f"✓ Metrics saved: {METRICS_PATH.resolve()}")

    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    df_features = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    df_features.to_csv(FEATURES_PATH, index=False)
    print(f"✓ Feature importances saved: {FEATURES_PATH.resolve()}")

    plot_feature_importance(data=df_features, save_path=str(FIGURE_PATH), figsize=(10, 10))
    
    dataframe_to_latex_table(
        df=df_report,
        save_path=str(TABLES_DIR / 'rf_baseline_metrics.tex'),
        caption="Performance Metrics for the Baseline Random Forest Classifier on the Validation Set.",
        label="tab:rf-baseline-metrics"
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Step 07: Random Forest Baseline Completed in {duration:.2f} seconds ---")

if __name__ == '__main__':
    main()