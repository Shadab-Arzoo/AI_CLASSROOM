"""
ExamGuard - High-Fidelity Multi-Scene Sample Video Generator
Creates a realistic 720p HD video clip in videos/sample_exam.mp4:
  - Scene 1 (Desk Anomaly): Candidate desk view with exam booklet, permitted stationery,
    and a photorealistic smartphone sliding onto the desk and being concealed.
  - Scene 2 (Hall CCTV): Authentic wide-angle classroom CCTV monitoring with multiple
    candidate desks, backpacks, books, and invigilator activity.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VIDEOS_DIR

def create_sample_video(
    output_path: Path = VIDEOS_DIR / "sample_exam.mp4",
    fps: int = 20
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 1. Load Phone Texture
    phone_path = VIDEOS_DIR / "extracted_phone.jpg"
    phone_img = None
    if phone_path.exists():
        phone_img = cv2.imread(str(phone_path))
        if phone_img is not None:
            # Scale to natural desk phone dimensions
            phone_img = cv2.resize(phone_img, (180, 240))

    # 2. Load CCTV classroom footage
    cctv_path = VIDEOS_DIR / "vidssave.com School class room video by IP 3MP CCTV CAMERA 2 WAY AUDIO 360P.mp4"
    cctv_cap = None
    if cctv_path.exists():
        cctv_cap = cv2.VideoCapture(str(cctv_path))

    # ── Phase 1: Close-up Candidate Desk Perspective (Frames 0 to 180, 9 seconds) ──
    total_desk_frames = 180

    for i in range(total_desk_frames):
        # Realistic desk background (warm natural oak desk tone with subtle grain)
        frame = np.full((height, width, 3), (188, 202, 215), dtype=np.uint8)

        # Subtle woodgrain noise
        np.random.seed(42 + (i % 5))
        grain = np.random.normal(0, 3.5, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        # Desk mat / boundary
        cv2.rectangle(frame, (60, 60), (1220, 680), (175, 190, 202), -1)
        cv2.rectangle(frame, (60, 60), (1220, 680), (150, 165, 178), 2)

        # Candidate Desk Roll Tag
        cv2.rectangle(frame, (80, 80), (480, 130), (245, 245, 248), -1)
        cv2.rectangle(frame, (80, 80), (480, 130), (180, 180, 190), 1)
        cv2.putText(frame, "DESK #14 | ROLL: 21CS-A104", (95, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        cv2.putText(frame, "CSE-AIML FINAL EXAMINATION", (95, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

        # Allowed Stationery: Wooden Pencil
        # Pencil Body (Yellow hexagon)
        cv2.rectangle(frame, (140, 260), (158, 520), (35, 180, 230), -1)
        cv2.line(frame, (149, 260), (149, 520), (25, 150, 200), 2)
        # Pencil Sharpened Tip
        pts_tip = np.array([[140, 260], [158, 260], [149, 230]], np.int32)
        cv2.fillPoly(frame, [pts_tip], (180, 210, 225))
        pts_graphite = np.array([[145, 245], [153, 245], [149, 230]], np.int32)
        cv2.fillPoly(frame, [pts_graphite], (40, 40, 40))
        # Brass ferrule & pink eraser on pencil top
        cv2.rectangle(frame, (140, 500), (158, 510), (100, 160, 190), -1)
        cv2.rectangle(frame, (140, 510), (158, 535), (170, 170, 235), -1)
        cv2.putText(frame, "Pencil", (125, 555), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

        # Allowed Stationery: White Eraser
        cv2.rectangle(frame, (190, 340), (270, 410), (250, 250, 252), -1)
        cv2.rectangle(frame, (190, 340), (235, 410), (160, 100, 50), -1) # Blue sleeve
        cv2.rectangle(frame, (190, 340), (270, 410), (160, 160, 160), 1)
        cv2.putText(frame, "Eraser", (200, 432), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

        # Allowed Stationery: Clear 15cm Ruler
        cv2.rectangle(frame, (110, 160), (460, 195), (230, 235, 240), -1)
        cv2.rectangle(frame, (110, 160), (460, 195), (180, 190, 200), 1)
        for rx in range(120, 450, 10):
            tick_h = 10 if (rx % 50 == 0) else 5
            cv2.line(frame, (rx, 160), (rx, 160 + tick_h), (120, 120, 120), 1)
        cv2.putText(frame, "Ruler (15cm)", (220, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

        # Examination Answer Booklet & Question Paper
        # Shadow
        cv2.rectangle(frame, (340, 145), (780, 635), (150, 160, 170), -1)
        # White paper sheets
        cv2.rectangle(frame, (330, 140), (770, 630), (253, 252, 250), -1)
        cv2.rectangle(frame, (330, 140), (770, 630), (190, 190, 190), 1)

        # Paper Header Banner
        cv2.rectangle(frame, (330, 140), (770, 185), (242, 244, 248), -1)
        cv2.putText(frame, "OFFICIAL UNIVERSITY EXAM ANSWER SCRIPT", (360, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
        cv2.putText(frame, "INSTRUCTIONS: NO ELECTRONIC DEVICES PERMITTED", (370, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 40, 40), 1)

        # Ruled notebook lines with simulated handwriting answers
        for ly in range(215, 610, 26):
            cv2.line(frame, (355, ly), (745, ly), (220, 222, 228), 1)
            # Simulated answer writing up to current progress
            if ly < 460:
                line_len = min(370, int((ly - 200) * 1.5))
                cv2.line(frame, (360, ly - 4), (360 + line_len, ly - 4), (80, 70, 60), 1)

        # Watermark Stamp
        cv2.circle(frame, (710, 220), 26, (200, 210, 235), 2)
        cv2.putText(frame, "VERIFIED", (690, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 180, 210), 1)

        # ── Simulated Incident: Real Smartphone slides onto desk (Frame 35+) ──
        if phone_img is not None and i >= 35:
            ph, pw = phone_img.shape[:2]

            # Smooth sliding motion from bottom onto the desk
            # Start: y=680, end: y=380, x=820
            t = min(1.0, (i - 35) / 35.0)
            eased_t = 1 - (1 - t)**3 # Ease out cubic
            current_x = int(820 + (1 - eased_t) * 40)
            current_y = int(680 - eased_t * 300)

            # In frames 130-180, candidate shifts phone slightly to hide under paper
            if i >= 130:
                conceal_t = min(1.0, (i - 130) / 35.0)
                current_x = int(820 - conceal_t * 90)

            # Natural drop shadow for phone
            cv2.ellipse(
                frame,
                (current_x + pw // 2 + 6, current_y + ph // 2 + 10),
                (pw // 2 + 10, ph // 2 + 10),
                0, 0, 360,
                (140, 150, 160), -1
            )

            # Composite real phone image
            if current_y + ph <= height and current_x + pw <= width:
                frame[current_y : current_y + ph, current_x : current_x + pw] = phone_img

        # Status Bar / Timestamp
        sec = i // fps
        cv2.rectangle(frame, (0, 0), (width, 36), (15, 15, 18), -1)
        cv2.putText(frame, f"CAM-01 · EXAM HALL B · 10:15:{sec:02d} AM · ASSISTIVE AI MONITORING", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 200), 1)

        out.write(frame)

    # ── Phase 2: Authentic Examination Hall CCTV Footage (Frames 180 to 380, 10 seconds) ──
    if cctv_cap is not None and cctv_cap.isOpened():
        cctv_frames = 190
        for i in range(cctv_frames):
            ret, cctv_frame = cctv_cap.read()
            if not ret:
                cctv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, cctv_frame = cctv_cap.read()
            if not ret:
                break

            # Scale CCTV footage to 1280x720 HD
            cctv_resized = cv2.resize(cctv_frame, (width, height))

            # CCTV Header Overlay
            sec = 9 + (i // fps)
            cv2.rectangle(cctv_resized, (0, 0), (width, 36), (15, 15, 18), -1)
            cv2.putText(cctv_resized, f"CAM-02 · WIDE EXAM HALL CCTV · 10:15:{sec:02d} AM · MULTI-OBJECT SURVEILLANCE", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 200), 1)

            out.write(cctv_resized)

        cctv_cap.release()

    out.release()
    print(f"[+] Enhanced sample exam video successfully created at: {output_path}")

if __name__ == "__main__":
    create_sample_video()
