"""
ExamGuard - Multi-Object Temporal Tracking Engine v2
=====================================================
Improvements:
  - Temporal confidence accumulation: detections must exceed threshold
    in 3 of last 5 frames before being confirmed (reduces flickers)
  - Stronger label voting with weighted recent frames
  - Confidence smoothing across frames (prevents single-frame spikes)
"""

import time
from collections import Counter
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from src.config import (
    CLASS_CONFIG,
    MIN_DETECTION_FRAMES,
    DEFAULT_ABSENCE_MONITOR_ENABLED,
    DEFAULT_ABSENCE_GRACE_SECONDS
)
from src.alerts import alert_manager
from src.detector import compute_iou
from src.policy import policy_manager


class ObjectTracker:
    def __init__(self, detector):
        self.detector = detector
        self.next_track_id = 1
        # Active tracks: {track_id: {track state dict}}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.max_lost_frames = 12  # Reduced from 15 — stale ghosts disappear faster
        self.history_window = 9    # Wider window for more stable label consensus
        self.confidence_window = 5 # Window for confidence smoothing

        # ── Candidate Absence Monitoring ──
        self.absence_enabled = DEFAULT_ABSENCE_MONITOR_ENABLED
        self.absence_grace_seconds = DEFAULT_ABSENCE_GRACE_SECONDS
        self.last_seen_person_time = time.time()
        self.is_absent = False
        self.absence_duration = 0.0
        self.last_absence_alert_time = 0.0
        self.last_person_count = 0

    def track_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "ExamHall-Cam-01",
        conf: float = 0.25
    ) -> Dict[str, Any]:
        """
        Runs dual-head object detection and associates detections across frames
        with temporal voting, confidence smoothing, and temporal accumulation.
        """
        if frame is None:
            return {"annotated_frame": frame, "tracks": [], "alerts": []}

        h, w = frame.shape[:2]
        diag = np.sqrt(w * w + h * h)

        # 1. Run Ensemble Detection
        detections = self.detector.detect(frame, conf=conf)

        # 2. Associate Detections with Active Tracks
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_track_ids = list(self.active_tracks.keys())

        if self.active_tracks and detections:
            cost_matrix = np.zeros((len(detections), len(unmatched_track_ids)))
            for d_idx, det in enumerate(detections):
                db = det["bbox"]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0

                for t_idx, trk_id in enumerate(unmatched_track_ids):
                    trk = self.active_tracks[trk_id]
                    tb = trk["bbox"]
                    tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0

                    iou = compute_iou(db, tb)
                    dist = np.sqrt((dcx - tcx)**2 + (dcy - tcy)**2) / max(1.0, diag)

                    score = iou
                    if dist < 0.14:
                        score += (0.14 - dist) * 2.5

                    if det["class_name"] == trk["class_name"]:
                        score += 0.25

                    cost_matrix[d_idx, t_idx] = score

            # Greedy bipartite matching
            used_detections = set()
            used_tracks = set()
            sorted_indices = np.dstack(
                np.unravel_index(np.argsort(-cost_matrix, axis=None), cost_matrix.shape)
            )[0]

            for d_idx, t_idx in sorted_indices:
                if d_idx in used_detections or t_idx in used_tracks:
                    continue
                if cost_matrix[d_idx, t_idx] < 0.14:
                    break

                trk_id = unmatched_track_ids[t_idx]
                det = detections[d_idx]
                old_trk = self.active_tracks[trk_id]

                # Exponential smoothing on bounding box
                smoothed_box = [
                    int(0.70 * det["bbox"][i] + 0.30 * old_trk["bbox"][i])
                    for i in range(4)
                ]

                # Update label history
                old_trk["history"].append(det["class_name"])
                if len(old_trk["history"]) > self.history_window:
                    old_trk["history"].pop(0)

                # Update confidence history for smoothing
                old_trk["conf_history"].append(det["confidence"])
                if len(old_trk["conf_history"]) > self.confidence_window:
                    old_trk["conf_history"].pop(0)

                # Weighted majority class vote (recent frames count more)
                weights = list(range(1, len(old_trk["history"]) + 1))
                weighted_counts: Dict[str, float] = {}
                for i, cls in enumerate(old_trk["history"]):
                    weighted_counts[cls] = weighted_counts.get(cls, 0.0) + weights[i]
                voted_class = max(weighted_counts, key=weighted_counts.get)

                # Smoothed confidence (average over recent frames)
                smoothed_conf = float(np.mean(old_trk["conf_history"]))

                eval_res = policy_manager.evaluate_item(voted_class)

                # Update track state
                old_trk["bbox"] = smoothed_box
                old_trk["confidence"] = smoothed_conf
                old_trk["class_name"] = voted_class
                old_trk["label"] = eval_res["label"]
                old_trk["severity"] = eval_res["severity"]
                old_trk["color"] = eval_res["color"]
                old_trk["allowed"] = eval_res["allowed"]
                old_trk["enabled"] = eval_res["enabled"]
                old_trk["frames_seen"] += 1
                old_trk["lost_frames"] = 0

                used_detections.add(d_idx)
                used_tracks.add(t_idx)

                if eval_res["enabled"]:
                    det_with_track = {
                        "track_id": trk_id,
                        "bbox": smoothed_box,
                        "class_name": voted_class,
                        "label": eval_res["label"],
                        "severity": eval_res["severity"],
                        "allowed": eval_res["allowed"],
                        "color": eval_res["color"],
                        "confidence": smoothed_conf,
                        "description": det.get("description", "")
                    }
                    matched_tracks.append(det_with_track)

            unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
            unmatched_track_ids = [unmatched_track_ids[i] for i in range(len(unmatched_track_ids)) if i not in used_tracks]

        # 3. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            eval_res = policy_manager.evaluate_item(det["class_name"])
            new_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[new_id] = {
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "label": eval_res["label"],
                "severity": eval_res["severity"],
                "allowed": eval_res["allowed"],
                "enabled": eval_res["enabled"],
                "confidence": det["confidence"],
                "color": eval_res["color"],
                "description": det.get("description", ""),
                "history": [det["class_name"]],
                "conf_history": [det["confidence"]],
                "frames_seen": 1,
                "lost_frames": 0
            }

            if eval_res["enabled"]:
                det_with_track = dict(det)
                det_with_track["track_id"] = new_id
                det_with_track["allowed"] = eval_res["allowed"]
                det_with_track["color"] = eval_res["color"]
                det_with_track["label"] = eval_res["label"]
                det_with_track["severity"] = eval_res["severity"]
                matched_tracks.append(det_with_track)

        # 4. Handle lost tracks
        for trk_id in list(unmatched_track_ids):
            self.active_tracks[trk_id]["lost_frames"] += 1
            if self.active_tracks[trk_id]["lost_frames"] > self.max_lost_frames:
                del self.active_tracks[trk_id]
            elif self.active_tracks[trk_id]["frames_seen"] >= MIN_DETECTION_FRAMES:
                lost_trk = self.active_tracks[trk_id]
                eval_res = policy_manager.evaluate_item(lost_trk["class_name"])
                if eval_res["enabled"]:
                    # Fade confidence for lost tracks
                    fade_factor = max(0.3, 1.0 - (lost_trk["lost_frames"] * 0.08))
                    matched_tracks.append({
                        "track_id": trk_id,
                        "class_name": lost_trk["class_name"],
                        "label": eval_res["label"],
                        "severity": eval_res["severity"],
                        "allowed": eval_res["allowed"],
                        "confidence": lost_trk["confidence"] * fade_factor,
                        "bbox": lost_trk["bbox"],
                        "color": eval_res["color"],
                        "description": lost_trk.get("description", "")
                    })

        # 5. Filter confirmed tracks for alert triggers
        # Temporal accumulation: only alert for tracks seen in >= MIN_DETECTION_FRAMES
        confirmed_alerts = []
        for t in matched_tracks:
            trk_id = t.get("track_id")
            track_data = self.active_tracks.get(trk_id, {})
            frames_seen = track_data.get("frames_seen", 1)

            # Must be confirmed across multiple frames AND not allowed
            if frames_seen >= MIN_DETECTION_FRAMES and not t.get("allowed", False):
                # Additional check: smoothed confidence must be above threshold
                if t.get("confidence", 0) >= 0.28:
                    confirmed_alerts.append(t)

        # 6. Trigger alerts for prohibited items
        new_alerts = alert_manager.process_detections(frame, confirmed_alerts, camera_id)

        # ── Candidate Absence Engine ──
        now = time.time()
        person_count = getattr(self.detector, "last_person_count", 0)
        self.last_person_count = person_count

        if person_count > 0:
            self.last_seen_person_time = now
            self.is_absent = False
            self.absence_duration = 0.0
        else:
            self.absence_duration = max(0.0, now - self.last_seen_person_time)
            if self.absence_enabled and self.absence_duration >= self.absence_grace_seconds:
                self.is_absent = True
                # Trigger absence alert every 8.0 seconds
                if now - self.last_absence_alert_time >= 8.0:
                    self.last_absence_alert_time = now
                    absence_alert = alert_manager.trigger_custom_alert(
                        frame=frame,
                        camera_id=camera_id,
                        class_name="candidate_absence",
                        label="Candidate Absent / Left Desk",
                        severity="HIGH",
                        confidence=0.95,
                        bbox=[10, 10, w - 10, h - 10],
                        notes=f"No candidate detected at exam desk for {self.absence_duration:.1f}s (Grace limit: {self.absence_grace_seconds}s)"
                    )
                    if absence_alert:
                        new_alerts.append(absence_alert)

        # 7. Draw boxes and HUD
        annotated_frame = self.detector.annotate(frame, detections=[], tracks=matched_tracks)

        # 8. Render Candidate Absence Banner if Active
        if self.absence_enabled:
            if self.is_absent:
                # Flashing Crimson Red Banner
                banner_h = 36
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 32), (w, 32 + banner_h), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                alert_text = f"[!] CRITICAL: CANDIDATE ABSENT / EMPTY DESK ({self.absence_duration:.1f}s)"
                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (20, 32 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            elif person_count == 0 and self.absence_duration > 1.0:
                # Amber Grace Warning Banner
                banner_h = 30
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 32), (w, 32 + banner_h), (0, 140, 255), -1)
                cv2.addWeighted(overlay, 0.80, annotated_frame, 0.20, 0, annotated_frame)
                grace_left = max(0.0, self.absence_grace_seconds - self.absence_duration)
                warning_text = f"[!] CANDIDATE NOT DETECTED | Grace: {grace_left:.1f}s remaining"
                cv2.putText(
                    annotated_frame,
                    warning_text,
                    (20, 32 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (10, 10, 10),
                    2,
                    cv2.LINE_AA
                )

        return {
            "annotated_frame": annotated_frame,
            "tracks": matched_tracks,
            "alerts": new_alerts,
            "absence": {
                "enabled": self.absence_enabled,
                "is_absent": self.is_absent,
                "duration": round(self.absence_duration, 1),
                "absence_duration": round(self.absence_duration, 1),
                "person_count": self.last_person_count,
                "grace_seconds": self.absence_grace_seconds
            }
        }
