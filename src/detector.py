st"""
ExamGuard - Precision Dual-Head Ensemble YOLO Detection Engine v3.1
==================================================================
Key Principles:
  1. High-Precision Threat Detection (YOLO11s):
     - Mobile Phones, Laptops, Tablets, Smartwatches, Calculators, Books, Bags.
     - High-confidence YOLO predictions (conf >= 0.45) are respected and preserved.
     - Size gates are realistic: phones/calculators held close to camera can occupy
       up to 65% of the frame without being falsely rejected.
  2. Targeted Disambiguation:
     - Laptop vs Book: Uses keyboard co-occurrence, aspect ratio (portrait vs landscape),
       and surface reflectance.
     - Remote -> Calculator: Evaluates button density and aspect ratio.
     - Low-confidence Phone -> Calculator: Only reclassifies when conf < 0.48 AND
       physical calculator keypad patterns are clearly present.
  3. Custom Stationery Engine (best.pt):
     - Pencils, Pens, Erasers, Sharpeners, Rulers.
     - Geometric sanity checks prevent book text lines from becoming false pencils.
  4. Real-time CPU Performance:
     - Single-pass 640px inference with CLAHE illumination enhancement for indoor webcams.
     - Priority Non-Maximum Suppression with cross-class overlap resolution.
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
from src.policy import policy_manager


# ──────────────────────────────────────────────────────────────
# Geometry Helpers
# ──────────────────────────────────────────────────────────────

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


def compute_area_ratio(bbox: List[int], frame_w: int, frame_h: int) -> float:
    """Computes the fraction of frame area occupied by the bounding box."""
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    return (bw * bh) / float(max(1, frame_w * frame_h))


def containment_ratio(inner: List[int], outer: List[int]) -> float:
    """What fraction of inner box's area lies inside outer box."""
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / float(inner_area)


# ──────────────────────────────────────────────────────────────
# Frame Enhancement
# ──────────────────────────────────────────────────────────────

def auto_enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Adaptive illumination normalization: CLAHE + Gamma.
    Improves detection on dark electronics and faint desk stationery in ~2ms.
    """
    if frame is None or frame.size == 0:
        return frame

    ch, cw = frame.shape[:2]
    sample = frame[ch // 4: 3 * ch // 4, cw // 4: 3 * cw // 4]
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_sample))

    # Low light / dim indoor lighting
    if mean_brightness < 110.0:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clip = 2.5 if mean_brightness < 80 else 1.8
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge([l_chan, a_chan, b_chan])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if mean_brightness < 70.0:
            gamma = 1.35
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)

        return enhanced

    return frame


# ──────────────────────────────────────────────────────────────
# Visual Analysis for Disambiguation
# ──────────────────────────────────────────────────────────────

def analyze_crop_features(crop: np.ndarray) -> Dict[str, Any]:
    """
    Photometric and texture features to distinguish paper/books from electronic screens.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return {
            "mean_luminance": 128.0,
            "std_luminance": 20.0,
            "is_light_paper": False,
            "is_dark_electronic": False,
            "white_pixel_ratio": 0.0,
            "has_text_lines": False,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    # White / light paper ratio (pixel intensity > 185)
    white_ratio = float(np.sum(gray > 185)) / float(gray.size)

    # Horizontal edges (characteristic of lined exam paper/books)
    horizontal_edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    horiz_ratio = float(np.count_nonzero(np.abs(horizontal_edges) > 30)) / float(max(1, gray.size))
    has_text_lines = horiz_ratio > 0.12 and white_ratio > 0.30

    is_light_paper = (mean_lum > 140.0) or (white_ratio > 0.45) or has_text_lines
    is_dark_electronic = (mean_lum < 90.0) and (white_ratio < 0.15)

    return {
        "mean_luminance": mean_lum,
        "std_luminance": std_lum,
        "is_light_paper": is_light_paper,
        "is_dark_electronic": is_dark_electronic,
        "white_pixel_ratio": white_ratio,
        "has_text_lines": has_text_lines,
    }


def is_calculator_crop(crop: np.ndarray) -> bool:
    """
    Detects whether a crop exhibits physical calculator characteristics:
    - LCD display in upper 30%
    - Dense grid of physical buttons in lower 70%
    - Aspect ratio typically 1.25 to 2.4
    """
    if crop is None or crop.shape[0] < 28 or crop.shape[1] < 20:
        return False

    h, w = crop.shape[:2]
    aspect = float(h) / float(max(1, w))
    if aspect < 1.15 or aspect > 2.6:
        return False

    # Lower 65% contains button rows
    lower = crop[int(h * 0.35):, :]
    gray_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lower, 50, 150)
    lower_edge_ratio = float(np.count_nonzero(edges)) / float(max(1, edges.size))

    # Upper 35% contains LCD display
    upper = crop[:int(h * 0.35), :]
    gray_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    upper_edges = cv2.Canny(gray_upper, 50, 150)
    upper_edge_ratio = float(np.count_nonzero(upper_edges)) / float(max(1, upper_edges.size))

    # Check for horizontal button rows
    lh, lw = gray_lower.shape
    horizontal_proj = np.sum(edges, axis=1) / float(max(1, lw))
    row_peaks = np.sum(horizontal_proj > 20.0)

    # Physical calculators have distinct button rows and higher edge density in keypad
    return (lower_edge_ratio > 0.08 and
            lower_edge_ratio >= upper_edge_ratio * 1.2 and
            row_peaks >= 3)


# ──────────────────────────────────────────────────────────────
# Detector Engine
# ──────────────────────────────────────────────────────────────

class Detector:
    def __init__(self, model_path: Optional[str] = None):
        self.custom_model = None
        self.threat_model = None
        self.is_custom_model = False
        self.model = None
        self.last_persons: List[Dict[str, Any]] = []
        self.last_person_count: int = 0
        self._load_engines(model_path)

    def _load_engines(self, custom_path: Optional[str] = None):
        """Loads both specialized YOLO models."""
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
        Runs dual-head ensemble detection:
        1. Threat model (YOLO11s) at 640px
        2. Disambiguation (Laptop vs Book, Remote -> Calculator, etc.)
        3. Stationery model (best.pt) at 640px
        4. Cross-class conflict resolution & Priority NMS
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]

        # Step 0: Fast adaptive illumination enhancement
        enhanced_frame = auto_enhance_frame(frame)

        all_raw_threats = []
        keyboards_detected = []
        persons_detected = []

        # ── Head 1: Threat Detection Engine (YOLO11s) ──
        if self.threat_model is not None:
            threat_conf = max(0.24, conf)
            threat_results = self.threat_model.predict(
                source=enhanced_frame,
                conf=threat_conf,
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

                    if raw_name == "keyboard":
                        keyboards_detected.append(bbox)

                    if raw_name == "person" and conf_val >= 0.30:
                        persons_detected.append({"bbox": bbox, "confidence": conf_val})

                    mapped = self._map_threat_class(raw_name)
                    if mapped:
                        area_ratio = compute_area_ratio(bbox, w, h)
                        if self._size_sanity_check(mapped, area_ratio, bbox, w, h):
                            all_raw_threats.append({
                                "class_name": mapped,
                                "raw_name": raw_name,
                                "confidence": conf_val,
                                "bbox": bbox
                            })

            self.last_persons = persons_detected
            self.last_person_count = len(persons_detected)

        # Scene-level dynamic thresholding: if abnormally many detections, raise bar
        if len(all_raw_threats) > 12:
            all_raw_threats = [d for d in all_raw_threats if d["confidence"] >= 0.40]
        elif len(all_raw_threats) > 8:
            all_raw_threats = [d for d in all_raw_threats if d["confidence"] >= 0.32]

        # Disambiguate threats (Book vs Laptop vs Phone vs Calculator)
        refined_threats = self._disambiguate_threats(frame, all_raw_threats, keyboards_detected)

        # ── Head 2: Custom Stationery Engine (best.pt) ──
        stationery_detections = []
        if self.custom_model is not None:
            stat_conf = max(0.30, conf)
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

                    if self._is_valid_stationery(raw_name, bbox, w, h, refined_threats, frame):
                        stationery_detections.append({
                            "class_name": raw_name,
                            "confidence": conf_val,
                            "bbox": bbox
                        })

        # Priority NMS and cross-class resolution
        final_detections = self._resolve_conflicts_and_nms(refined_threats, stationery_detections)
        return final_detections

    def _map_threat_class(self, raw_name: str) -> Optional[str]:
        """Maps COCO classes into ExamGuard domains."""
        name = raw_name.replace(" ", "_")
        MAPPING = {
            "cell_phone": "cell phone",
            "mobile_phone": "cell phone",
            "phone": "cell phone",
            "remote": "calculator",
            "laptop": "laptop",
            "computer": "laptop",
            "tv": "tablet",
            "monitor": "tablet",
            "keyboard": "keyboard",
            "mouse": "mouse",
            "book": "book",
            "clock": "clock",
            "backpack": "backpack",
            "handbag": "handbag",
            "suitcase": "handbag",
            "scissors": "scissors",
        }
        return MAPPING.get(name)

    def _size_sanity_check(
        self,
        class_name: str,
        area_ratio: float,
        bbox: List[int],
        frame_w: int,
        frame_h: int
    ) -> bool:
        """
        Rejects absurd detections (covering entire screen or tiny 1-2px noise).
        Allows close-up objects (e.g. phone held to camera up to 65% of frame).
        """
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        min_dim = min(bw, bh)

        # Reject boxes covering >75% of frame (background/desk)
        if area_ratio > 0.75:
            return False

        # Reject boxes smaller than 12px or < 0.1% area (noise)
        if min_dim < 12 or area_ratio < 0.001:
            return False

        # Clocks/smartwatches on wrist should not be huge wall clocks
        if class_name == "clock" and area_ratio > 0.06:
            return False

        # Keyboards / mice
        if class_name == "mouse" and area_ratio > 0.08:
            return False

        return True

    def _disambiguate_threats(
        self,
        frame: np.ndarray,
        threats: List[Dict[str, Any]],
        keyboards: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Disambiguates between visually similar items:
        - Cell Phone vs Calculator vs Book
        - Laptop vs Book vs Tablet
        """
        refined = []
        h, w = frame.shape[:2]

        for item in threats:
            cls_name = item["class_name"]
            raw_name = item.get("raw_name", "")
            bbox = item["bbox"]
            conf = item["confidence"]
            x1, y1, x2, y2 = bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_ratio = bw / float(bh)  # > 1.0 landscape, < 1.0 portrait

            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            props = analyze_crop_features(crop)

            # Keyboard presence check
            has_keyboard = False
            for kbox in keyboards:
                if compute_iou(bbox, kbox) > 0.15:
                    has_keyboard = True
                    break
                kcx, kcy = (kbox[0] + kbox[2]) // 2, (kbox[1] + kbox[3]) // 2
                if x1 <= kcx <= x2 and y1 <= kcy <= y2:
                    has_keyboard = True
                    break

            # ── 1. Cell Phone Disambiguation ──
            if cls_name == "cell phone":
                # If YOLO is confident (>= 0.45), respect cell phone!
                # Only check calculator if confidence is marginal or it exhibits calculator features
                if conf < 0.48 and is_calculator_crop(crop):
                    cls_name = "calculator"

            # ── 2. Calculator Disambiguation (COCO "remote") ──
            elif cls_name == "calculator":
                if raw_name == "remote":
                    # Remote mapped to calculator. If it's a sleek smartphone with high aspect ratio, keep check
                    # Otherwise in exam setting, remote is classified as calculator
                    pass

            # ── 3. Laptop Disambiguation ──
            elif cls_name == "laptop":
                # Laptops are landscape. If it's a tall portrait rectangle without keyboard, it's a book
                if not has_keyboard:
                    if aspect_ratio < 0.82 and props["is_light_paper"]:
                        cls_name = "book"
                        conf = max(conf, 0.48)
                    elif props["is_light_paper"] and props["has_text_lines"] and aspect_ratio < 1.1:
                        cls_name = "book"
                        conf = max(conf, 0.45)

            # ── 4. Book Disambiguation ──
            elif cls_name == "book":
                # If a keyboard is inside the book box, it's a laptop
                if has_keyboard:
                    cls_name = "laptop"
                    conf = max(conf, 0.50)
                elif props["is_dark_electronic"] and aspect_ratio > 1.25 and conf < 0.50:
                    cls_name = "laptop"

            # ── 5. Tablet Disambiguation ──
            elif cls_name == "tablet":
                if has_keyboard:
                    cls_name = "laptop"
                elif props["is_light_paper"] and not props["is_dark_electronic"]:
                    cls_name = "book"

            # Evaluate against active exam hall policy rules
            eval_res = policy_manager.evaluate_item(cls_name)
            if not eval_res["enabled"]:
                continue

            refined.append({
                "class_name": cls_name,
                "label": eval_res["label"],
                "severity": eval_res["severity"],
                "allowed": eval_res["allowed"],
                "status_badge": eval_res["status_badge"],
                "confidence": conf,
                "bbox": bbox,
                "color": eval_res["color"],
                "description": CLASS_CONFIG.get(cls_name, {}).get("description", "")
            })

        return refined

    def _is_valid_stationery(
        self,
        class_name: str,
        bbox: List[int],
        frame_w: int,
        frame_h: int,
        threat_boxes: List[Dict[str, Any]],
        frame: np.ndarray = None
    ) -> bool:
        """
        Validates stationery items from custom best.pt model.
        Prevents ruled text lines on paper from triggering false pencil detections.
        """
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        frame_area = frame_w * frame_h
        ratio = area / float(frame_area)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        min_dim = min(bw, bh)

        # Check containment inside a book or paper
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        inside_book = False
        for t in threat_boxes:
            if t.get("class_name") in ("book", "paper"):
                tx1, ty1, tx2, ty2 = t["bbox"]
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    inside_book = True
                    break

        if class_name == "pencil":
            # Real pencils are slender
            if aspect < 3.2 or ratio > 0.07 or min_dim < 4:
                return False

            # If inside a book, require stricter aspect to avoid text lines
            if inside_book and aspect < 5.0:
                return False

            # Color check: text lines have very low saturation; real pencils often have color
            if frame is not None and inside_book:
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0 and crop.shape[0] >= 3 and crop.shape[1] >= 3:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    sat = float(np.mean(hsv[:, :, 1]))
                    val = float(np.mean(hsv[:, :, 2]))
                    if sat < 18 and val > 160:
                        return False

        elif class_name in ["eraser", "sharpener"]:
            if ratio > 0.035 or max(bw, bh) > 130 or min_dim < 8:
                return False
            if inside_book and ratio < 0.003:
                return False

        elif class_name == "ruler":
            if aspect < 2.4 or ratio > 0.12:
                return False

        return True

    def _resolve_conflicts_and_nms(
        self,
        threats: List[Dict[str, Any]],
        stationery: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cross-class conflict resolution and priority NMS:
        1. Threat detections take precedence.
        2. Overlapping duplicate boxes of the same class (IoU > 0.20 or containment > 0.35)
           are suppressed, keeping the higher confidence detection.
        3. Overlapping boxes of different classes keep the higher confidence detection (IoU > 0.35).
        4. Stationery items that don't conflict with threats are preserved.
        """
        # ── Step 1: NMS within threat items ──
        sorted_threats = sorted(threats, key=lambda d: d["confidence"], reverse=True)
        kept_threats = []
        while sorted_threats:
            best = sorted_threats.pop(0)
            kept_threats.append(best)
            survivors = []
            for candidate in sorted_threats:
                iou = compute_iou(best["bbox"], candidate["bbox"])
                cont_best = containment_ratio(best["bbox"], candidate["bbox"])
                cont_cand = containment_ratio(candidate["bbox"], best["bbox"])
                max_cont = max(cont_best, cont_cand)

                # Same class duplicate suppression
                if candidate["class_name"] == best["class_name"]:
                    if iou < 0.20 and max_cont < 0.35:
                        survivors.append(candidate)
                else:
                    # Different class overlap resolution
                    if iou < 0.35 and max_cont < 0.50:
                        survivors.append(candidate)

            sorted_threats = survivors

        final_list = list(kept_threats)

        # ── Step 2: Add stationery items that do not conflict ──
        for stat in stationery:
            eval_res = policy_manager.evaluate_item(stat["class_name"])
            if not eval_res["enabled"]:
                continue

            conflicts = False
            for t in kept_threats:
                iou = compute_iou(stat["bbox"], t["bbox"])
                contained = containment_ratio(stat["bbox"], t["bbox"])
                if iou > 0.25 or contained > 0.70:
                    conflicts = True
                    break

            if not conflicts:
                final_list.append({
                    "class_name": stat["class_name"],
                    "label": eval_res["label"],
                    "severity": eval_res["severity"],
                    "allowed": eval_res["allowed"],
                    "status_badge": eval_res["status_badge"],
                    "confidence": stat["confidence"],
                    "bbox": stat["bbox"],
                    "color": eval_res["color"],
                    "description": CLASS_CONFIG.get(stat["class_name"], {}).get("description", "")
                })

        return final_list

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        tracks: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Draws aesthetic bounding boxes, badges, and alerts with permission-aware highlights.
        """
        annotated = frame.copy()
        items = tracks if tracks is not None else detections

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            is_allowed = item.get("allowed", False)
            severity = item.get("severity", "INFO")
            if severity == "ALLOWED":
                is_allowed = True

            # Mint Green for allowed, Red/Amber for prohibited
            color = item.get("color", (60, 214, 63) if is_allowed else (0, 0, 255))
            label = item.get("label", item.get("class_name", "Object"))
            conf = item.get("confidence", 0.0)
            track_id = item.get("track_id")

            # Draw bounding box
            thickness = 2
            if not is_allowed and severity == "HIGH":
                thickness = 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Build label text with permission tag
            status_text = "[ALLOWED]" if is_allowed else "[PROHIBITED]"
            tag = f"{label} {status_text} {int(conf * 100)}%"
            if track_id is not None:
                tag = f"#{track_id} {tag}"

            # Label badge background
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 10, max(22, y1)), color, -1)

            # Text color
            text_color = (10, 10, 10) if is_allowed else (255, 255, 255)
            cv2.putText(
                annotated,
                tag,
                (x1 + 5, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                text_color,
                1,
                cv2.LINE_AA
            )

        return annotated
