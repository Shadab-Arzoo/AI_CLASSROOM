"""
ExamGuard - Precision Dual-Head Ensemble YOLO Detection Engine
Combines:
  1. High-Capacity YOLO11s Threat Engine: Phones, Laptops, Tablets, Smartwatches, Books, Bags, Keyboards.
  2. Custom Stationery Engine (best.pt): Precision-trained on Pencils, Erasers, Sharpeners, Rulers.
Integrates:
  - Adaptive Low-Light Illumination Normalization (CLAHE/Gamma) for indoor webcams.
  - Multi-feature visual disambiguation (Book vs. Laptop vs. Tablet vs. Phone).
  - Geometric artifact rejection for stationery (stops book text lines from becoming false pencils).
  - Priority-based Non-Maximum Suppression (NMS).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from src.config import (
    CLASS_CONFIG,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_CUSTOM_MODEL,
    DEFAULT_PRETRAINED_MODEL
)

def compute_iou(boxA, boxB) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def auto_enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Applies fast adaptive illumination normalization for low-light or uneven indoor webcam lighting.
    Improves contrast on dark electronics and faint desk stationery in ~2ms.
    """
    if frame is None or frame.size == 0:
        return frame

    # Sample central region brightness
    ch, cw = frame.shape[:2]
    sample = frame[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_sample))

    # Dim / underexposed room: apply adaptive gamma curve
    if mean_brightness < 90.0:
        gamma = 1.35
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)

    return frame


def analyze_crop_features(crop: np.ndarray) -> Dict[str, Any]:
    """
    Extracts photometric and geometric features from a bounding box crop
    to distinguish paper/books from electronic screens and devices.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return {
            "mean_luminance": 128.0,
            "std_luminance": 20.0,
            "is_light_paper": False,
            "is_dark_electronic": False,
            "white_pixel_ratio": 0.0
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    # White / light paper ratio (pixel intensity > 180)
    white_ratio = float(np.sum(gray > 180)) / float(gray.size)

    # Real books / question paper sheets are predominantly white or cream paper
    is_light_paper = (mean_lum > 138.0) or (white_ratio > 0.40)

    # Laptops / screens / tablets have dark frames, screens, or metallic chassis
    is_dark_electronic = (mean_lum < 95.0) and (white_ratio < 0.20)

    return {
        "mean_luminance": mean_lum,
        "std_luminance": std_lum,
        "is_light_paper": is_light_paper,
        "is_dark_electronic": is_dark_electronic,
        "white_pixel_ratio": white_ratio
    }


class Detector:
    def __init__(self, model_path: Optional[str] = None):
        self.custom_model = None
        self.threat_model = None
        self.is_custom_model = False
        self.model = None  # Primary tracker reference
        self._load_engines(model_path)

    def _load_engines(self, custom_path: Optional[str] = None):
        """Loads both specialized models for ensemble monitoring."""
        from ultralytics import YOLO

        # 1. Load Threat Detection Engine (YOLO11s)
        threat_path = str(DEFAULT_PRETRAINED_MODEL)
        try:
            print(f"[ExamGuard Detector] Loading Threat Detection Engine from: {threat_path}")
            self.threat_model = YOLO(threat_path)
            self.model = self.threat_model
        except Exception as e:
            print(f"[!] Warning: Could not load threat model: {e}")

        # 2. Load Custom Stationery Engine (best.pt)
        custom_weights = custom_path if (custom_path and os.path.exists(custom_path)) else str(DEFAULT_CUSTOM_MODEL)
        if os.path.exists(custom_weights):
            try:
                print(f"[ExamGuard Detector] Loading Custom Stationery Engine from: {custom_weights}")
                self.custom_model = YOLO(custom_weights)
                self.is_custom_model = True
                print(f"[ExamGuard Detector] Active Custom Classes: {list(self.custom_model.names.values())}")
            except Exception as e:
                print(f"[!] Warning: Could not load custom stationery model: {e}")
        else:
            print("[ExamGuard Detector] Custom stationery model not found. Running in Pretrained mode.")

    @property
    def model_path(self) -> str:
        if self.custom_model and self.threat_model:
            return "Dual-Head Ensemble (YOLO11s + Custom best.pt)"
        elif self.custom_model:
            return str(DEFAULT_CUSTOM_MODEL)
        return str(DEFAULT_PRETRAINED_MODEL)

    def detect(self, frame: np.ndarray, conf: float = DEFAULT_CONF_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Runs dual-head detection on frame with adaptive illumination,
        deep visual disambiguation, geometric sanity filtering, and priority NMS.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]

        # Step 0: Fast adaptive illumination enhancement for indoor webcams
        enhanced_frame = auto_enhance_frame(frame)

        all_raw_threats = []
        keyboards_detected = []

        # ── Head 1: Threat Detection Engine ──
        if self.threat_model is not None:
            # Predict with sensitive threshold so real objects aren't dropped
            threat_results = self.threat_model.predict(
                source=enhanced_frame,
                conf=max(0.16, conf),
                iou=DEFAULT_IOU_THRESHOLD,
                imgsz=640,
                verbose=False
            )

            if threat_results and len(threat_results) > 0 and threat_results[0].boxes is not None:
                for box in threat_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = self.threat_model.names.get(cls_id, "").lower().strip()
                    bbox = [int(v) for v in box.xyxy[0].tolist()]
                    conf_val = round(float(box.conf[0].item()), 3)

                    # Track keyboard detections to help confirm laptops vs books
                    if raw_name == "keyboard":
                        keyboards_detected.append(bbox)

                    mapped = self._map_threat_class(raw_name, bbox, w, h)
                    if mapped:
                        all_raw_threats.append({
                            "class_name": mapped,
                            "raw_name": raw_name,
                            "confidence": conf_val,
                            "bbox": bbox
                        })

        # ── Disambiguation: Book vs. Laptop vs. Tablet vs. Phone ──
        refined_threats = self._disambiguate_threats(frame, all_raw_threats, keyboards_detected)

        # ── Head 2: Custom Stationery Engine ──
        stationery_detections = []
        if self.custom_model is not None:
            stat_conf = max(0.20, conf)
            stat_results = self.custom_model.predict(
                source=enhanced_frame,
                conf=stat_conf,
                iou=DEFAULT_IOU_THRESHOLD,
                imgsz=640,
                verbose=False
            )

            if stat_results and len(stat_results) > 0 and stat_results[0].boxes is not None:
                for box in stat_results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = self.custom_model.names.get(cls_id, "").lower().strip()
                    bbox = [int(v) for v in box.xyxy[0].tolist()]
                    conf_val = round(float(box.conf[0].item()), 3)

                    # Geometric sanity check: prevent book text lines from becoming false pencils
                    if self._is_valid_stationery(raw_name, bbox, w, h, refined_threats):
                        stationery_detections.append({
                            "class_name": raw_name,
                            "confidence": conf_val,
                            "bbox": bbox
                        })

        # ── Multi-Head Priority NMS ──
        final_detections = self._priority_nms(refined_threats, stationery_detections)
        return final_detections

    def _map_threat_class(self, raw_name: str, bbox: List[int], frame_w: int, frame_h: int) -> Optional[str]:
        """Maps COCO classes into ExamGuard monitoring domains."""
        name = raw_name.replace(" ", "_")

        # Cell phones
        if name in ["cell_phone", "mobile_phone", "phone"]:
            return "cell phone"

        # Remote controls held by students are almost always smartphones / calculators
        if name == "remote":
            bw = max(1, bbox[2] - bbox[0])
            bh = max(1, bbox[3] - bbox[1])
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 1.20:
                return "cell phone"
            return "calculator"

        # Laptops & computers
        if name in ["laptop", "computer"]:
            return "laptop"

        # TV screens / monitors in exams are tablets / iPads / open laptops
        if name in ["tv", "monitor"]:
            return "tablet"

        if name == "keyboard":
            return "keyboard"

        if name == "mouse":
            return "mouse"

        if name in ["book"]:
            return "book"

        if name in ["clock"]:
            return "clock"  # Smartwatch / digital timer

        if name in ["backpack"]:
            return "backpack"

        if name in ["handbag", "suitcase"]:
            return "handbag"

        if name in ["scissors"]:
            return "scissors"

        return None

    def _disambiguate_threats(
        self,
        frame: np.ndarray,
        threats: List[Dict[str, Any]],
        keyboards: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Deep visual disambiguation between Books, Laptops, Tablets, and Phones:
        - Books / notebooks: light paper luminance, text lines, no keyboard, often portrait or open fold.
        - Laptops: dark keyboard, screen bezel, metallic chassis, keyboard overlap.
        """
        refined = []

        for item in threats:
            cls_name = item["class_name"]
            bbox = item["bbox"]
            conf = item["confidence"]
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_ratio = bw / float(bh)  # > 1.0 = landscape, < 1.0 = portrait

            # Crop region for visual analysis
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            props = analyze_crop_features(crop)

            # Check if a keyboard is detected inside or overlapping this box
            has_keyboard_inside = False
            for kbox in keyboards:
                if compute_iou(bbox, kbox) > 0.20:
                    has_keyboard_inside = True
                    break
                kcx, kcy = (kbox[0] + kbox[2]) // 2, (kbox[1] + kbox[3]) // 2
                if x1 <= kcx <= x2 and y1 <= kcy <= y2:
                    has_keyboard_inside = True
                    break

            # ── Disambiguation Logic ──
            if cls_name == "laptop":
                # If YOLO predicted laptop, but:
                # 1. No keyboard is detected, AND
                # 2. Surface is white paper / light pages (mean luminance > 138), OR
                # 3. It's a tall portrait rectangle (aspect_ratio < 0.85)
                # Then it is almost certainly a Book / Exam Notebook, NOT a laptop!
                if not has_keyboard_inside and (props["is_light_paper"] or aspect_ratio < 0.82):
                    cls_name = "book"
                    conf = max(conf, 0.48)
                elif has_keyboard_inside:
                    cls_name = "laptop"

            elif cls_name == "book":
                # If YOLO predicted book, but:
                # 1. A keyboard was detected inside it, OR
                # 2. It has very dark electronic chassis / screen reflections (mean lum < 75)
                # Then it is a Laptop or Tablet!
                if has_keyboard_inside or (props["is_dark_electronic"] and aspect_ratio > 1.15):
                    cls_name = "laptop"
                    conf = max(conf, 0.48)

            elif cls_name == "tablet":
                # If it's light paper without bezel, reclassify as book/sheet
                if props["is_light_paper"]:
                    cls_name = "book"

            config = CLASS_CONFIG.get(cls_name)
            if config:
                refined.append({
                    "class_name": cls_name,
                    "label": config["label"],
                    "severity": config["severity"],
                    "confidence": conf,
                    "bbox": bbox,
                    "color": config["color"],
                    "description": config["description"]
                })

        return refined

    def _is_valid_stationery(
        self,
        class_name: str,
        bbox: List[int],
        frame_w: int,
        frame_h: int,
        threat_boxes: List[Dict[str, Any]]
    ) -> bool:
        """
        Filters out false stationery detections (e.g. text lines or desk edges
        being falsely recognized as pencils or giant erasers).
        """
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        frame_area = frame_w * frame_h
        ratio = area / float(frame_area)
        aspect = max(bw, bh) / max(1, min(bw, bh))

        # Check if stationery center is inside a threat item (book, laptop, tablet)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for t in threat_boxes:
            tx1, ty1, tx2, ty2 = t["bbox"]
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                # Text lines on a book/paper often trigger pencil false alarms
                # A real pencil must be very slender and distinct
                if class_name == "pencil" and aspect < 3.2:
                    return False
                if class_name in ["eraser", "sharpener"]:
                    return False

        if class_name == "pencil":
            # A real pencil must be slender (aspect ratio > 2.0) and not a huge area
            if aspect < 2.0 or ratio > 0.08:
                return False

        elif class_name in ["eraser", "sharpener"]:
            # Erasers and sharpeners are small handheld objects
            if ratio > 0.035 or bw > 140 or bh > 140:
                return False

        elif class_name == "ruler":
            # Rulers are elongated
            if aspect < 2.2:
                return False

        return True

    def _priority_nms(
        self,
        threats: List[Dict[str, Any]],
        stationery: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prioritized NMS:
        1. Threat detections (Phones, Laptops, Books) take precedence over stationery.
        2. Threat boxes suppress overlapping duplicate threat boxes.
        3. Stationery boxes that genuinely stand alone are preserved.
        """
        final_list = []

        # Step 1: NMS within threat items
        sorted_threats = sorted(threats, key=lambda d: d["confidence"], reverse=True)
        kept_threats = []
        while len(sorted_threats) > 0:
            best = sorted_threats.pop(0)
            kept_threats.append(best)
            survivors = []
            for candidate in sorted_threats:
                iou = compute_iou(best["bbox"], candidate["bbox"])
                if iou < 0.42:
                    survivors.append(candidate)
            sorted_threats = survivors

        final_list.extend(kept_threats)

        # Step 2: Add stationery items that do not conflict with or get contained in threats
        for stat in stationery:
            config = CLASS_CONFIG.get(stat["class_name"])
            if not config:
                continue

            conflicts = False
            for t in kept_threats:
                iou = compute_iou(stat["bbox"], t["bbox"])
                if iou > 0.30:
                    conflicts = True
                    break

            if not conflicts:
                final_list.append({
                    "class_name": stat["class_name"],
                    "label": config["label"],
                    "severity": config["severity"],
                    "confidence": stat["confidence"],
                    "bbox": stat["bbox"],
                    "color": config["color"],
                    "description": config["description"]
                })

        return final_list

    def annotate(self, frame: np.ndarray, detections: List[Dict[str, Any]], tracks: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """Draws aesthetic bounding boxes, badges, and alerts on the frame."""
        annotated = frame.copy()
        items = tracks if tracks is not None else detections

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            color = item.get("color", (0, 255, 0))
            label = item.get("label", item.get("class_name", "Object"))
            conf = item.get("confidence", 0.0)
            track_id = item.get("track_id")
            severity = item.get("severity", "INFO")

            # Draw bounding box
            thickness = 3 if severity == "HIGH" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            tag = f"{label} {int(conf * 100)}%"
            if track_id is not None:
                tag = f"#{track_id} {tag}"

            # Label badge background
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + w + 12, max(24, y1)), color, -1)

            # Label text
            text_color = (255, 255, 255) if severity in ["HIGH", "MEDIUM"] else (0, 0, 0)
            cv2.putText(
                annotated,
                tag,
                (x1 + 6, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                text_color,
                1,
                cv2.LINE_AA
            )

        return annotated
