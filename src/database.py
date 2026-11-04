"""
ExamGuard - Database Module (SQLite)
Handles persistent storage of exam alerts, evidence links, and human invigilator review states.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import DB_PATH

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    object_class TEXT NOT NULL,
                    object_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity TEXT NOT NULL,
                    track_id INTEGER,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    evidence_path TEXT,
                    review_status TEXT DEFAULT 'PENDING',
                    invigilator_notes TEXT DEFAULT '',
                    reviewed_at TEXT,
                    reviewed_by TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (review_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity)")
            conn.commit()

    def insert_alert(self, alert_data: Dict[str, Any]) -> str:
        """Inserts a new alert record and returns the generated alert_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (
                    alert_id, timestamp, camera_id, object_class, object_label,
                    confidence, severity, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    evidence_path, review_status, invigilator_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_data["alert_id"],
                alert_data.get("timestamp", datetime.datetime.now().isoformat()),
                alert_data.get("camera_id", "ExamHall-Cam-01"),
                alert_data["object_class"],
                alert_data.get("object_label", alert_data["object_class"]),
                float(alert_data["confidence"]),
                alert_data.get("severity", "MEDIUM"),
                alert_data.get("track_id"),
                alert_data.get("bbox", [0, 0, 0, 0])[0],
                alert_data.get("bbox", [0, 0, 0, 0])[1],
                alert_data.get("bbox", [0, 0, 0, 0])[2],
                alert_data.get("bbox", [0, 0, 0, 0])[3],
                alert_data.get("evidence_path", ""),
                alert_data.get("review_status", "PENDING"),
                alert_data.get("invigilator_notes", "")
            ))
            conn.commit()
            return alert_data["alert_id"]

    def get_alerts(self, limit: int = 50, offset: int = 0, status: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch alerts with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if status:
                query += " AND review_status = ?"
                params.append(status)
            if severity:
                query += " AND severity = ?"
                params.append(severity)
                
            query += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single alert details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_review(self, alert_id: str, review_status: str, notes: str, reviewer: str = "Invigilator") -> bool:
        """Update the human review status and notes for an alert."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts
                SET review_status = ?,
                    invigilator_notes = ?,
                    reviewed_at = ?,
                    reviewed_by = ?
                WHERE alert_id = ?
            """, (
                review_status,
                notes,
                datetime.datetime.now().isoformat(),
                reviewer,
                alert_id
            ))
            conn.commit()
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate stats for invigilator dashboard cards."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE review_status = 'PENDING'")
            pending_reviews = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'HIGH'")
            high_severity_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE review_status = 'FLAGGED_INFRACTION'")
            verified_infractions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE review_status = 'DISMISSED_FALSE_ALARM'")
            dismissed_false_alarms = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT object_label, COUNT(*) as count 
                FROM alerts 
                GROUP BY object_label 
                ORDER BY count DESC
            """)
            class_counts = {row["object_label"]: row["count"] for row in cursor.fetchall()}
            
            return {
                "total_alerts": total_alerts,
                "pending_reviews": pending_reviews,
                "high_severity_count": high_severity_count,
                "verified_infractions": verified_infractions,
                "dismissed_false_alarms": dismissed_false_alarms,
                "class_counts": class_counts
            }

    def clear_all_alerts(self) -> int:
        """Deletes all alert records, resets table sequence, and clears statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM alerts")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='alerts'")
            conn.commit()
            return count

# Singleton instance
db = Database()

