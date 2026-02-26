import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "podcastiq.db"

def get_db_connection():
    """Create a database connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with the schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Session/Upload Management
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS podcast_uploads (
        uploadid INTEGER PRIMARY KEY AUTOINCREMENT,
        source VARCHAR(500) UNIQUE,
        title VARCHAR(500),
        thumbnail_url TEXT,
        status VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Full Processing Results (Unified for easier retrieval)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processing_results (
        resultid INTEGER PRIMARY KEY AUTOINCREMENT,
        uploadid INTEGER REFERENCES podcast_uploads(uploadid) ON DELETE CASCADE,
        transcript TEXT,
        summaries_json TEXT,
        qa_pairs_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_upload_source ON podcast_uploads(source)')
    
    conn.commit()
    conn.close()

def get_cached_result(source: str) -> Optional[Dict[str, Any]]:
    """Check if a source has already been processed and return results."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT u.title, u.thumbnail_url, r.summaries_json, r.qa_pairs_json 
        FROM podcast_uploads u
        JOIN processing_results r ON u.uploadid = r.uploadid
        WHERE u.source = ? AND u.status = 'completed'
    ''', (source,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "title": row["title"],
            "thumbnailUrl": row["thumbnail_url"],
            "overallSummary": json.loads(row["summaries_json"]),
            "qaPairs": json.loads(row["qa_pairs_json"]),
            "cached": True
        }
    return None

def save_processing_result(source: str, title: str, thumbnail_url: str, transcript: str, summaries: Dict, qa_pairs: List[Dict]):
    """Save the final processing results to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert or update upload record
        cursor.execute('''
            INSERT INTO podcast_uploads (source, title, thumbnail_url, status)
            VALUES (?, ?, ?, 'completed')
            ON CONFLICT(source) DO UPDATE SET 
                title=excluded.title, 
                thumbnail_url=excluded.thumbnail_url,
                status='completed'
        ''', (source, title, thumbnail_url))
        
        upload_id = cursor.lastrowid
        if not upload_id:
            cursor.execute('SELECT uploadid FROM podcast_uploads WHERE source = ?', (source,))
            upload_id = cursor.fetchone()[0]
            
        # Insert processing result
        cursor.execute('''
            INSERT INTO processing_results (uploadid, transcript, summaries_json, qa_pairs_json)
            VALUES (?, ?, ?, ?)
        ''', (upload_id, transcript, json.dumps(summaries), json.dumps(qa_pairs)))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_history() -> List[Dict[str, Any]]:
    """Get a list of all processed podcasts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT uploadid, source, title, thumbnail_url, created_at 
        FROM podcast_uploads 
        WHERE status = 'completed'
        ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
