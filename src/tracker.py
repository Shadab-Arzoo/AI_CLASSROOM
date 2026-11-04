"""
ExamGuard - Multi-Object Temporal Tracking Engine with Sliding Window Voting
Maintains persistent IDs across video frames, tracks object dwell time,
implements temporal label voting (stops label flickering between Book and Laptop),
and coordinates with AlertManager.
Uses combined IoU and spatial center-distance association with hysteresis.
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import CLASS_CONFIG, MIN_DETECTION_FRAMES
from src.alerts import alert_manager
from src.detector import compute_iou

class ObjectTracker:
    def __init__(self, detector):
        self.detector = detector
        self.next_track_id = 1
        # Active tracks: {track_id: {"bbox": ..., "class_name": ..., "history": [...], "frames_seen": ..., "lost_frames": ...}}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.max_lost_frames = 15  # Keep track alive for up to 15 lost frames to prevent flickering
        self.history_window = 7    # Sliding window for majority label voting

    def track_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "ExamHall-Cam-01",
        conf: float = 0.20
    ) -> Dict[str, Any]:
        """
        Runs dual-head object detection and associates detections across frames with temporal voting.
        Returns annotated frame, active track list, and any newly triggered alerts.
        """
        if frame is None:
            return {"annotated_frame": frame, "tracks": [], "alerts": []}

        h, w = frame.shape[:2]
        diag = np.sqrt(w * w + h * h)

        # 1. Run Ensemble Detection (Stationery + Unauthorized Devices)
        detections = self.detector.detect(frame, conf=conf)

        # 2. Associate Detections with Active Tracks using IoU + Center Distance Fallback
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_track_ids = list(self.active_tracks.keys())

        if self.active_tracks and detections:
            # Build cost matrix
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

                    # Spatial affinity score: IoU + center proximity
                    score = iou
                    if dist < 0.14:  # within 14% of screen
                        score += (0.14 - dist) * 2.5

                    # Bonus for matching current consensus class
                    if det["class_name"] == trk["class_name"]:
                        score += 0.25

                    cost_matrix[d_idx, t_idx] = score

            # Greedy bipartite matching
            used_detections = set()
            used_tracks = set()

            sorted_indices = np.dstack(np.unravel_index(np.argsort(-cost_matrix, axis=None), cost_matrix.shape))[0]

            for d_idx, t_idx in sorted_indices:
                if d_idx in used_detections or t_idx in used_tracks:
                    continue
                # Lower threshold so moved/tilted objects don't lose tracking
                if cost_matrix[d_idx, t_idx] < 0.14:
                    break

                trk_id = unmatched_track_ids[t_idx]
                det = detections[d_idx]
                old_trk = self.active_tracks[trk_id]

                # Exponential smoothing on bounding box to stop visual jitter
                smoothed_box = [
                    int(0.70 * det["bbox"][i] + 0.30 * old_trk["bbox"][i])
                    for i in range(4)
                ]

                # Update label history and compute majority consensus
                old_trk["history"].append(det["class_name"])
                if len(old_trk["history"]) > self.history_window:
                    old_trk["history"].pop(0)

                # Majority class vote eliminates frame-to-frame label swapping
                class_counts = Counter(old_trk["history"])
                voted_class, _ = class_counts.most_common(1)[0]
                config = CLASS_CONFIG.get(voted_class, CLASS_CONFIG.get(det["class_name"]))

                # Update track state
                old_trk["bbox"] = smoothed_box
                old_trk["confidence"] = det["confidence"]
                old_trk["class_name"] = voted_class
                if config:
                    old_trk["label"] = config["label"]
                    old_trk["severity"] = config["severity"]
                    old_trk["color"] = config["color"]
                old_trk["frames_seen"] += 1
                old_trk["lost_frames"] = 0

                used_detections.add(d_idx)
                used_tracks.add(t_idx)

                det_with_track = {
                    "track_id": trk_id,
                    "bbox": smoothed_box,
                    "class_name": voted_class,
                    "label": old_trk["label"],
                    "severity": old_trk["severity"],
                    "color": old_trk["color"],
                    "confidence": det["confidence"],
                    "description": det.get("description", "")
                }
                matched_tracks.append(det_with_track)

            unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
            unmatched_track_ids = [unmatched_track_ids[i] for i in range(len(unmatched_track_ids)) if i not in used_tracks]

        # 3. Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[new_id] = {
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "label": det["label"],
                "severity": det["severity"],
                "confidence": det["confidence"],
                "color": det["color"],
                "description": det.get("description", ""),
                "history": [det["class_name"]],
                "frames_seen": 1,
                "lost_frames": 0
            }

            det_with_track = dict(det)
            det_with_track["track_id"] = new_id
            matched_tracks.append(det_with_track)

        # 4. Handle lost tracks (smoothing: keep showing temporarily if missed for a frame or two)
        for trk_id in list(unmatched_track_ids):
            self.active_tracks[trk_id]["lost_frames"] += 1
            if self.active_tracks[trk_id]["lost_frames"] > self.max_lost_frames:
                del self.active_tracks[trk_id]
            elif self.active_tracks[trk_id]["frames_seen"] >= 2:
                # Add smoothed track to visualization so box doesn't flicker away
                lost_trk = self.active_tracks[trk_id]
                matched_tracks.append({
                    "track_id": trk_id,
                    "class_name": lost_trk["class_name"],
                    "label": lost_trk["label"],
                    "severity": lost_trk["severity"],
                    "confidence": lost_trk["confidence"],
                    "bbox": lost_trk["bbox"],
                    "color": lost_trk["color"],
                    "description": lost_trk.get("description", "")
                })

        # 5. Filter confirmed tracks for alert triggers
        confirmed_alerts = []
        for t in matched_tracks:
            trk_id = t.get("track_id")
            frames_seen = self.active_tracks.get(trk_id, {}).get("frames_seen", 1)
            if frames_seen >= MIN_DETECTION_FRAMES:
                confirmed_alerts.append(t)

        # 6. Trigger alerts for prohibited items
        new_alerts = alert_manager.process_detections(frame, confirmed_alerts, camera_id)

        # 7. Draw boxes and HUD
        annotated_frame = self.detector.annotate(frame, detections=[], tracks=matched_tracks)

        return {
            "annotated_frame": annotated_frame,
            "tracks": matched_tracks,
            "alerts": new_alerts
        }
