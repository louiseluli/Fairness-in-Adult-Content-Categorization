# -*- coding: utf-8 -*-
"""
09_bert_baseline.py

Purpose:
    Trains, evaluates, and saves a more advanced baseline classifier using a
    pre-trained DistilBERT model. This script establishes a state-of-the-art
    performance benchmark to compare against the Random Forest model and for
    later bias mitigation experiments.
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

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
MODEL_DIR = OUTPUT_DIR / 'models' / 'bert_baseline'
PREDICTIONS_PATH = OUTPUT_DIR / 'data' / 'bert_baseline_val_predictions.csv'
METRICS_PATH = OUTPUT_DIR / 'data' / 'bert_baseline_val_metrics.csv'
TABLES_DIR = Path(CONFIG['project']['root']) / 'dissertation' / 'auto_tables'

# --- 2. Data Preparation ---

def prepare_data(df: pd.DataFrame, target_col: str = 'categories', top_n_classes: int = 10):
    """Prepares and encodes data for the BERT model."""
    print("Preparing data for BERT modeling...")
    df['primary_category'] = df[target_col].str.split(',').str[0].str.strip()
    top_classes = df['primary_category'].value_counts().nlargest(top_n_classes).index
    df_filtered = df[df['primary_category'].isin(top_classes)].copy()
    
    le = LabelEncoder()
    df_filtered['labels'] = le.fit_transform(df_filtered['primary_category'])
    
    print(f"✓ Data prepared. Target variable is 'primary_category', focusing on top {top_n_classes} classes.")
    return df_filtered, le

# --- 3. Main Execution ---

def main():
    start_time = time.monotonic()
    print("--- Starting Step 09: BERT Baseline ---")

    # --- GPU/MPS Acceleration Check ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Apple M-series GPU (MPS) is available and will be used.")
    else:
        device = torch.device("cpu")
        print("  - MPS device not found. Falling back to CPU. This will be slower.")

    # Load data
    df = pd.read_parquet(CORPUS_PATH)
    train_ids = pd.read_csv(TRAIN_IDS_PATH)['video_id']
    val_ids = pd.read_csv(VAL_IDS_PATH)['video_id']
    
    df, label_encoder = prepare_data(df)
    
    train_df = df[df['video_id'].isin(train_ids)]
    val_df = df[df['video_id'].isin(val_ids)]

    # Convert to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['model_input_text', 'labels']])
    val_dataset = Dataset.from_pandas(val_df[['model_input_text', 'labels']])

    # Initialize tokenizer and tokenize datasets
    print("Tokenizing datasets...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
         return tokenizer(examples['model_input_text'], padding='max_length', truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    print("✓ Tokenization complete.")
    
    # Initialize model
    num_labels = len(label_encoder.classes_)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)
    
    # Define training arguments with corrected parameter names
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / 'training_checkpoints'),
        num_train_epochs=1,
        per_device_train_batch_size=32, # Increased batch size for GPU
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch", # CORRECTED: Renamed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        use_mps_device=torch.backends.mps.is_available() # Explicitly enable MPS
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Fine-tune the model
    print("Fine-tuning DistilBERT model... (This may take a significant amount of time)")
    trainer.train()
    print("✓ Model fine-tuning complete.")
    
    # Evaluate the model
    print("Evaluating model on validation set...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    report = classification_report(val_dataset['labels'], y_pred, target_names=label_encoder.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    print("Validation Performance (rounded):\n", df_report.round(3))
    
    # Save artifacts
    print("\nSaving artefacts...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"✓ Model and tokenizer saved to: {MODEL_DIR.resolve()}")

    val_df_with_preds = val_df.copy()
    val_df_with_preds['predicted_category'] = label_encoder.inverse_transform(y_pred)
    val_df_with_preds[['video_id', 'primary_category', 'predicted_category']].to_csv(PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved: {PREDICTIONS_PATH.resolve()}")

    df_report.to_csv(METRICS_PATH)
    print(f"✓ Metrics saved: {METRICS_PATH.resolve()}")
    
    dataframe_to_latex_table(
        df=df_report,
        save_path=str(TABLES_DIR / 'bert_baseline_metrics.tex'),
        caption="Performance Metrics for the Baseline DistilBERT Classifier on the Validation Set.",
        label="tab:bert-baseline-metrics"
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"\n--- Step 09: BERT Baseline Completed in {duration/60:.2f} minutes ---")

if __name__ == '__main__':
    main()