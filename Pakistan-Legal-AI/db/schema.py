import sqlite3
import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from cryptography.fernet import Fernet

# Simple Regex patterns for PII redaction (Pakistan Context)
PII_PATTERNS = [
    (re.compile(r'\b\d{5}-\d{7}-\d{1}\b'), "[REDACTED_CNIC]"),
    (re.compile(r'\b(?:(?:\+92)|(?:0092)|(?:0))?3\d{2}-?\d{7}\b'), "[REDACTED_PHONE]"),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'), "[REDACTED_EMAIL]")
]

def redact_pii(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

class DatabaseSchema:
    def __init__(self, db_path: str = "evaluations.db"):
        self.db_path = db_path
        
        # In a real app, this key should come from a secure env var, e.g. os.getenv("ENCRYPTION_KEY")
        key_str = os.getenv("ENCRYPTION_KEY")
        if key_str:
            self.fernet = Fernet(key_str.encode())
        else:
            # Fallback for dev - generate a random one
            # Note: This means data encrypted in one run might not be readable in the next!
            # For production, MUST provide ENCRYPTION_KEY
            self.fernet = Fernet(Fernet.generate_key())
            
        self.init_db()
    
    def init_db(self):
        """Initialize the evaluation database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS eval_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    org_id TEXT,
                    question_encrypted BLOB NOT NULL,
                    answer_encrypted BLOB NOT NULL,
                    contexts_encrypted BLOB NOT NULL,
                    faithfulness REAL,
                    answer_relevance REAL,
                    context_recall REAL,
                    overall_score REAL
                )
            ''')
            conn.commit()

    def _encrypt(self, text: str) -> bytes:
        return self.fernet.encrypt(text.encode("utf-8"))

    def _decrypt(self, data: bytes) -> str:
        return self.fernet.decrypt(data).decode("utf-8")
    
    def log_evaluation(self, question: str, answer: str, contexts: List[str],
                      faithfulness: float, answer_relevance: float,
                      context_recall: float, user_id: Optional[str] = None,
                      org_id: Optional[str] = None) -> int:
        """Redact PII, encrypt, and log a new evaluation result."""
        if not org_id:
            raise ValueError("org_id is required for tenant-isolated evaluation logs")
        overall_score = (faithfulness + answer_relevance + context_recall) / 3.0
        
        # PII Redaction
        clean_q = redact_pii(question)
        clean_a = redact_pii(answer)
        clean_c = [redact_pii(c) for c in contexts]

        # Encryption
        enc_q = self._encrypt(clean_q)
        enc_a = self._encrypt(clean_a)
        enc_c = self._encrypt(json.dumps(clean_c))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO eval_logs 
                (question_encrypted, answer_encrypted, contexts_encrypted, faithfulness, answer_relevance, 
                 context_recall, overall_score, user_id, org_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (enc_q, enc_a, enc_c, faithfulness, 
                  answer_relevance, context_recall, overall_score, user_id, org_id))
            conn.commit()
            return cursor.lastrowid
    
    def get_rolling_averages(self, org_id: str) -> Dict[str, Dict[str, float]]:
        """Get rolling averages for different time periods."""
        if not org_id:
            raise ValueError("org_id is required")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            org_clause = "WHERE org_id = ?"
            org_params = [org_id]
            
            periods = {
                "7_days": datetime.now() - timedelta(days=7),
                "30_days": datetime.now() - timedelta(days=30),
                "all_time": datetime.min
            }
            
            results = {}
            
            for period_name, start_date in periods.items():
                if period_name == "all_time":
                    cursor.execute(f'''
                        SELECT 
                            AVG(faithfulness) as avg_faithfulness,
                            AVG(answer_relevance) as avg_answer_relevance,
                            AVG(context_recall) as avg_context_recall,
                            AVG(overall_score) as avg_overall_score,
                            COUNT(*) as total_evaluations,
                            SUM(CASE WHEN faithfulness < 0.7 THEN 1 ELSE 0 END) as low_faithfulness_count
                        FROM eval_logs 
                        {org_clause}
                    ''', org_params)
                else:
                    cursor.execute(f'''
                        SELECT 
                            AVG(faithfulness) as avg_faithfulness,
                            AVG(answer_relevance) as avg_answer_relevance,
                            AVG(context_recall) as avg_context_recall,
                            AVG(overall_score) as avg_overall_score,
                            COUNT(*) as total_evaluations,
                            SUM(CASE WHEN faithfulness < 0.7 THEN 1 ELSE 0 END) as low_faithfulness_count
                        FROM eval_logs 
                        {org_clause} AND timestamp >= ?
                    ''', org_params + [start_date.isoformat()])
                
                row = cursor.fetchone()
                if row and row[0]:  # If we have data
                    hallucination_rate = (row[5] / row[4]) * 100 if row[4] > 0 else 0
                    results[period_name] = {
                        "avg_faithfulness": round(row[0], 3),
                        "avg_answer_relevance": round(row[1], 3),
                        "avg_context_recall": round(row[2], 3),
                        "avg_overall_score": round(row[3], 3),
                        "total_evaluations": row[4],
                        "hallucination_rate": round(hallucination_rate, 2)
                    }
                else:
                    results[period_name] = {
                        "avg_faithfulness": 0.0,
                        "avg_answer_relevance": 0.0,
                        "avg_context_recall": 0.0,
                        "avg_overall_score": 0.0,
                        "total_evaluations": 0,
                        "hallucination_rate": 0.0
                    }
            
            return results
    
    def get_time_series_data(self, org_id: str, days: int = 30) -> List[Dict]:
        """Get daily aggregated data for time series visualization."""
        if not org_id:
            raise ValueError("org_id is required")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            org_clause = "AND org_id = ?"
            org_params = [org_id]
            
            cursor.execute(f'''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevance) as avg_answer_relevance,
                    AVG(context_recall) as avg_context_recall,
                    AVG(overall_score) as avg_overall_score,
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN faithfulness < 0.7 THEN 1 ELSE 0 END) as low_faithfulness_count
                FROM eval_logs 
                WHERE timestamp >= date('now', '-{days} days') {org_clause}
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            ''', org_params)
            
            rows = cursor.fetchall()
            time_series = []
            
            for row in rows:
                hallucination_rate = (row[6] / row[5]) * 100 if row[5] > 0 else 0
                time_series.append({
                    "date": row[0],
                    "avg_faithfulness": round(row[1], 3),
                    "avg_answer_relevance": round(row[2], 3),
                    "avg_context_recall": round(row[3], 3),
                    "avg_overall_score": round(row[4], 3),
                    "total_evaluations": row[5],
                    "hallucination_rate": round(hallucination_rate, 2)
                })
            
            return time_series

    def delete_by_org(self, org_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM eval_logs WHERE org_id = ?', (org_id,))
            conn.commit()
            return cursor.rowcount

    def delete_by_user(self, org_id: str, user_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM eval_logs WHERE org_id = ? AND user_id = ?', (org_id, user_id))
            conn.commit()
            return cursor.rowcount

    def delete_older_than(self, days: int) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM eval_logs WHERE timestamp < ?', (cutoff,))
            conn.commit()
            return cursor.rowcount

# Global instance for easy import
db_schema = DatabaseSchema()
