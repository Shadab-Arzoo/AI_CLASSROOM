"""
ExamGuard - Video Stream & Camera Manager
Manages video sources (Webcam or Video Files), runs real-time frame processing,
and yields JPEG bytes for MJPEG HTTP streaming.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Generator
import cv2
import numpy as np
from src.config import DEFAULT_CAMERA_ID, DEFAULT_VIDEO_SOURCE, DEFAULT_CONF_THRESHOLD, VIDEOS_DIR
from src.detector import Detector
from src.tracker import ObjectTracker

class VideoStreamManager:
    def __init__(self, detector: Detector, tracker: ObjectTracker):
        self.detector = detector
        self.tracker = tracker
        self.source = DEFAULT_VIDEO_SOURCE
        self.camera_id = DEFAULT_CAMERA_ID
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # State metrics
        self.current_fps = 0.0
        self.total_frames_processed = 0
        self.is_tracking_enabled = True
        self.confidence_threshold = DEFAULT_CONF_THRESHOLD

    def start_source(self, source: Optional[str] = None):
        """Starts or switches video capture source (webcam index or file path)."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()

            if source is None:
                source = self.source

            # Check if source is a digit string (webcam index)
            if isinstance(source, str) and source.isdigit():
                self.source = int(source)
            else:
                self.source = source

            print(f"[ExamGuard Video] Opening video source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)

            # If opening webcam or file failed, try fallback
            if not self.cap.isOpened():
                print(f"[ExamGuard Video] Warning: Unable to open source {self.source}.")
                # If we tried a webcam and failed, try checking sample videos
                sample_videos = list(VIDEOS_DIR.glob("*.mp4"))
                if sample_videos:
                    print(f"[ExamGuard Video] Falling back to sample video: {sample_videos[0]}")
                    self.source = str(sample_videos[0])
                    self.cap = cv2.VideoCapture(self.source)

            self.is_running = self.cap.isOpened()
            print(f"[ExamGuard Video] Video capture active: {self.is_running}")

    def stop(self):
        """Stops the video capture."""
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Reads a single frame from the active video capture."""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None

            ret, frame = self.cap.read()
            # If video file reached the end, loop back to the beginning
            if not ret and isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            return frame if ret else None

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Generator yielding MJPEG multipart frames for HTTP streaming."""
        if not self.is_running:
            self.start_source()

        last_time = time.time()
        fps_smoothing = 0.9

        while True:
            frame = self.get_frame()
            if frame is None:
                # If no frame available (e.g. no camera attached), generate synthetic status card
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "ExamGuard: No Camera/Video Source Connected",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
                cv2.putText(
                    blank,
                    "Upload an Exam MP4 video or select Webcam in Controls",
                    (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
                ret, jpeg = cv2.imencode('.jpg', blank)
                time.sleep(0.1)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                continue

            now = time.time()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                self.current_fps = fps_smoothing * self.current_fps + (1 - fps_smoothing) * current_fps

            # Process frame with tracking and alerts
            if self.is_tracking_enabled:
                result = self.tracker.track_frame(
                    frame=frame,
                    camera_id=self.camera_id,
                    conf=self.confidence_threshold
                )
                display_frame = result["annotated_frame"]
            else:
                detections = self.detector.detect(frame, conf=self.confidence_threshold)
                display_frame = self.detector.annotate(frame, detections)

            # Draw HUD watermark (Camera ID, FPS, Time)
            h, w = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, 0), (w, 32), (15, 15, 15), -1)
            hud_text = f"CAM: {self.camera_id} | FPS: {int(self.current_fps)} | MODEL: {'Custom' if self.detector.is_custom_model else 'Pretrained'} | CONF: {int(self.confidence_threshold * 100)}%"
            cv2.putText(display_frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

            self.total_frames_processed += 1

            ret, jpeg = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

