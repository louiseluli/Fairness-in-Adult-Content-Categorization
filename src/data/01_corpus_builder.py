# -*- coding: utf-8 -*-
"""
01_corpus_builder.py

Purpose:
    Constructs the master machine learning corpus from the raw SQLite database.
    This script is the foundational step for the entire analysis pipeline.
"""

import os
import re
import sys
import json
import sqlite3
import pandas as pd
import warnings # Import the warnings library
from pathlib import Path

# --- 1. Configuration and Setup ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.theme_manager import load_config
from src.utils.database import create_connection

CONFIG = load_config()

# Define paths from the new config structure
CORPUS_PATH = Path(CONFIG['paths']['data']) / 'ml_corpus.parquet'
STATS_PATH = Path(CONFIG['paths']['data']) / 'corpus_stats.json'
LEXICON_PATH = Path(CONFIG['paths']['config']) / 'protected_terms.json'

def _load_protected_terms():
    """Loads the protected terms lexicon."""
    with open(LEXICON_PATH, 'r') as f:
        return json.load(f)

PROTECTED_TERMS = _load_protected_terms()

# --- 2. Data Fetching ---

def fetch_data_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetches and joins the core data from the SQLite database."""
    print("Fetching data from database... This may take a moment.")
    query = f"""
    SELECT
        v.video_id, v.title, v.duration, v.views, v.rating, v.ratings, v.publish_date,
        GROUP_CONCAT(DISTINCT t.tag) AS tags,
        GROUP_CONCAT(DISTINCT c.category) AS categories
    FROM {CONFIG['db']['tables']['videos']} v
    LEFT JOIN {CONFIG['db']['tables']['video_tags']} t ON v.video_id = t.video_id
    LEFT JOIN {CONFIG['db']['tables']['video_categories']} c ON v.video_id = c.video_id
    GROUP BY v.video_id
    """
    try:
        df = pd.read_sql_query(query, conn)
        print(f"✓ Fetched {len(df):,} records from the database.")
        return df
    except Exception as e:
        print(f"✗ ERROR: Failed to fetch data. Reason: {e}")
        return pd.DataFrame()

# --- 3. Text Cleaning and Feature Engineering ---

def clean_text(text: str) -> str:
    """Applies basic text cleaning."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_protected_group_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Uses the protected terms lexicon to create binary indicator columns."""
    print("Creating protected group features...")
    df_out = df.copy()
    feature_keys = CONFIG['project_specifics']['feature_generation_keys']

    for group in feature_keys:
        categories = PROTECTED_TERMS.get(group, {})
        if not isinstance(categories, dict): continue
        for category, terms in categories.items():
            if isinstance(terms, list):
                col_name = f"{group}_{category}"
                valid_terms = [term for term in terms if isinstance(term, str)]
                if not valid_terms: continue
                pattern = r'\b(' + '|'.join(re.escape(term) for term in valid_terms) + r')\b'
                
                # --- FIX for UserWarning ---
                # This context manager suppresses the specific, benign warning from pandas.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    df_out[col_name] = df_out[text_col].str.contains(
                        pattern, regex=True, na=False
                    ).astype(int)
    
    print("✓ Protected group features created.")
    return df_out

def create_intersectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates intersectional features based on column names defined in settings.yaml."""
    print("Creating intersectional features...")
    df_out = df.copy()
    isect_config = CONFIG['project_specifics']['intersection']
    race_col, gender_col, output_col = isect_config['primary_race_col'], isect_config['primary_gender_col'], isect_config['output_col_name']

    if race_col in df_out.columns and gender_col in df_out.columns:
        df_out[output_col] = ((df_out[race_col] == 1) & (df_out[gender_col] == 1)).astype(int)
        print(f"✓ Created '{output_col}' feature.")
    else:
        print(f"✗ WARNING: Could not create '{output_col}' feature. Required columns not found.")
        
    return df_out

# --- 4. Main Execution Pipeline ---

def main():
    """Main function to build and save the corpus."""
    print("--- Starting Step 01: Corpus Builder ---")
    
    conn = create_connection()
    if conn is None:
        print("✗ Halting execution due to database connection failure.")
        return

    df = fetch_data_from_db(conn)
    conn.close()
    if df.empty: return

    print("Cleaning and combining text fields...")
    df['tags'] = df['tags'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['combined_text'] = df['title'] + ' ' + df['tags'] + ' ' + df['categories']
    df['combined_text_clean'] = df['combined_text'].apply(clean_text)
    df['model_input_text'] = (df['title'] + ' ' + df['tags']).apply(clean_text)
    
    df = create_protected_group_features(df, 'combined_text_clean')
    df = create_intersectional_features(df)
    
    print("Saving artefacts...")
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"✓ Artefact saved: {CORPUS_PATH.resolve()}")

    stats = {
        'num_records': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'protected_group_counts': {
            col: int(df[col].sum()) for col in df.columns if any(k in col for k in ['race_', 'gender_', 'intersectional_'])
        }
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✓ Artefact saved: {STATS_PATH.resolve()}")

    print("\n--- Step 01: Corpus Builder Completed Successfully ---")

if __name__ == '__main__':
    main()