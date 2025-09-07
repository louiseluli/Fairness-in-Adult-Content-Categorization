# msc_fairness_project/src/utils/database.py

import pandas as pd
import os
import sqlite3
from sqlite3 import Error
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Optional

# --- Didactic Explanation ---
# This script is our single source of truth for the database.
# It defines the path, creates the database file if it doesn't exist,
# and sets up the 'videos' table with the correct schema.
# By keeping this separate, our main collector script doesn't need to worry
# about the database structure, making our code more modular.
# ---

# Define the path to the database file. We place it in the top-level 'data' folder.
# os.path.join is used to create a path that works on any operating system (Windows, macOS, Linux).

# -----------------------------------------------------------------------------
# 1) Database path (override with env var if needed) + ensure ./data exists
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = Path(os.environ.get("MSC_DB_FILE", str(DATA_DIR / "redtube_videos.db"))).resolve()

# -----------------------------------------------------------------------------
# 2) Connection helper with safe defaults (WAL, timeouts, foreign keys, row factory)
# -----------------------------------------------------------------------------

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Better concurrency + durability trade-off
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    # Avoid 'database is locked' under parallel readers/writers
    cur.execute("PRAGMA busy_timeout=5000;")  # ms
    # Keep relational integrity if you use FKs (tags/categories tables)
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.close()


def create_connection(check_same_thread: bool = True) -> Optional[sqlite3.Connection]:
    """
    Create a database connection to the SQLite database.
    If the file does not exist, it will be created.
    """
    try:
        conn = sqlite3.connect(
            DB_FILE,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=check_same_thread,
        )
        # dict-like row access: row["title"]
        conn.row_factory = sqlite3.Row
        _apply_pragmas(conn)
        print(f"Successfully connected to SQLite database at {DB_FILE}")
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

@contextmanager
def get_conn(check_same_thread: bool = True) -> Iterator[sqlite3.Connection]:
    """
    Context manager for connections. Usage:
        with get_conn() as conn:
            ...
    Ensures connections are closed and transactions committed/rolled back.
    """
    conn = create_connection(check_same_thread=check_same_thread)
    if conn is None:
        raise RuntimeError("Failed to open database connection.")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# 3) Schema (normalized for fairness slicing + resumable collection)
#    - videos: main facts
#    - video_tags: 1:N tags (enables clean group/audit slices)
#    - video_categories: 1:N categories
#    - audit_terms: curated identity terms (race/gender/orientation)
#    - collection_state: track API daily cap + resume pointers
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    url             TEXT,
    duration        INTEGER,             -- seconds (prefer INTEGER over TEXT)
    views           INTEGER,
    rating          REAL,
    ratings         INTEGER,
    publish_date    TEXT,                -- ISO-8601 string (YYYY-MM-DD)
    category_source TEXT,
    is_active       INTEGER,             -- 0/1
    retrieved_at    TEXT NOT NULL        -- ISO-8601 UTC timestamp
);

CREATE TABLE IF NOT EXISTS video_tags (
    video_id    INTEGER NOT NULL,
    tag         TEXT    NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    UNIQUE(video_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_video_tags_tag ON video_tags(tag);

CREATE TABLE IF NOT EXISTS video_categories (
    video_id    INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    UNIQUE(video_id, category)
);

CREATE INDEX IF NOT EXISTS idx_video_categories_category ON video_categories(category);

CREATE TABLE IF NOT EXISTS audit_terms (
    term    TEXT PRIMARY KEY,
    group_name TEXT NOT NULL            -- e.g., 'race', 'gender', 'orientation'
);

CREATE TABLE IF NOT EXISTS collection_state (
    day                 TEXT PRIMARY KEY,    -- YYYY-MM-DD (UTC)
    requests_used       INTEGER NOT NULL DEFAULT 0,
    last_page_fetched   INTEGER NOT NULL DEFAULT 0,
    reset_at            TEXT NOT NULL        -- ISO-8601 UTC timestamp of next reset
);

-- Helpful read paths
CREATE INDEX IF NOT EXISTS idx_videos_publish_date ON videos(publish_date);
CREATE INDEX IF NOT EXISTS idx_videos_is_active ON videos(is_active);
"""

def load_data_from_db(db_path: Path) -> pd.DataFrame:
    """
    Connects to the SQLite DB and loads the full, current video data into a DataFrame.
    This function is now centralized here for reuse across all analysis and modeling scripts.
    
    Args:
        db_path (Path): The path to the SQLite database file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the joined video and tag data.
    """
    print(f"Connecting to database at: {db_path}...")
    try:
        con = sqlite3.connect(db_path)
        # This robust query joins videos with their aggregated tags.
        query = """
        SELECT
            v.*,
            t.tags
        FROM
            videos v
        LEFT JOIN
            (SELECT video_id, GROUP_CONCAT(tag) as tags FROM video_tags GROUP BY video_id) t
        ON v.video_id = t.video_id;
        """
        df = pd.read_sql_query(query, con)
        con.close()
        print(f"Successfully loaded {len(df):,} video records.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def create_tables(conn: sqlite3.Connection) -> None:
    try:
        conn.executescript(SCHEMA_SQL)
        print("Schema ensured: videos, video_tags, video_categories, audit_terms, collection_state.")
    except Error as e:
        print(f"Error creating tables: {e}")

# -----------------------------------------------------------------------------
# 4) Setup entrypoint (idempotent)
# -----------------------------------------------------------------------------
def setup_database() -> None:
    """Create the database and tables if needed."""
    with get_conn() as conn:
        create_tables(conn)

if __name__ == "__main__":
    setup_database()