"""
ExamGuard - Alert Management Engine
Generates structured alerts, manages cooldowns/persistence to avoid spam,
captures visual evidence frames, and saves alerts to the database.
"""

import time
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from src.config import EVIDENCE_DIR, ALERT_COOLDOWN_SECONDS, CLASS_CONFIG
from src.database import db

class AlertManager:
    def __init__(self):
        # Keeps track of {track_id or class_name: last_alert_time}
        self.cooldown_tracker: Dict[str, float] = {}
        # Keeps count of consecutive frames an object has been seen: {key: count}
        self.persistence_tracker: Dict[str, int] = {}

    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        camera_id: str = "ExamHall-Cam-01"
    ) -> List[Dict[str, Any]]:
        """
        Processes frame detections and generates alerts for unauthorized/suspicious objects.
        Returns list of newly triggered alerts for immediate UI broadcasting.
        """
        now = time.time()
        new_alerts = []

        for det in detections:
            # Authorized / permitted items must NEVER trigger violation alerts
            if det.get("allowed", False):
                continue

            severity = det.get("severity", "INFO")
            # Only trigger alerts for prohibited / unauthorized objects
            if severity not in ["HIGH", "MEDIUM", "PROHIBITED"]:
                continue

            class_name = det["class_name"]
            track_id = det.get("track_id")
            
            # Unique key for cooldown tracking
            key = f"{class_name}_{track_id}" if track_id is not None else class_name

            # Check cooldown: don't re-alert the same object track within cooldown period
            last_time = self.cooldown_tracker.get(key, 0.0)
            if now - last_time < ALERT_COOLDOWN_SECONDS:
                continue

            # Update cooldown timestamp
            self.cooldown_tracker[key] = now

            # Generate unique alert ID
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            short_id = uuid.uuid4().hex[:6].upper()
            alert_id = f"ALT-{timestamp_str}-{short_id}"

            # Save visual evidence frame
            evidence_filename = f"{alert_id}.jpg"
            evidence_path = EVIDENCE_DIR / evidence_filename
            self._save_evidence(frame, det, evidence_path, alert_id)

            alert_record = {
                "alert_id": alert_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "camera_id": camera_id,
                "object_class": class_name,
                "object_label": det.get("label", class_name),
                "confidence": det["confidence"],
                "severity": severity,
                "track_id": track_id,
                "bbox": det["bbox"],
                "evidence_path": f"evidence/{evidence_filename}",
                "review_status": "PENDING",
                "invigilator_notes": ""
            }

            # Persist to database
            db.insert_alert(alert_record)
            new_alerts.append(alert_record)
            print(f"[ExamGuard Alert] [!] {severity} ALERT: {det['label']} detected (Conf: {int(det['confidence']*100)}%) | ID: {alert_id}")

        return new_alerts

    def _save_evidence(self, frame: np.ndarray, det: Dict[str, Any], filepath: Path, alert_id: str):
        """Annotates and saves an evidence snapshot with timestamp and alert header."""
        evidence = frame.copy()
        x1, y1, x2, y2 = det["bbox"]
        color = det.get("color", (0, 0, 255))
        label = det.get("label", det["class_name"])
        conf = det.get("confidence", 0.0)

        # Highlight detection bounding box
        cv2.rectangle(evidence, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            evidence,
            f"{label} ({int(conf * 100)}%)",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

        # Overlay Banner at top of snapshot for documentation
        h, w = evidence.shape[:2]
        cv2.rectangle(evidence, (0, 0), (w, 40), (20, 20, 20), -1)
        banner_text = f"EXAMGUARD ALERT: {alert_id} | TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(evidence, banner_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(filepath), evidence)

# Singleton instance
alert_manager = AlertManager()

