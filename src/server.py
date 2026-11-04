"""
ExamGuard - FastAPI Web & Alert Server
Hosts the REST APIs, Web Dashboard, MJPEG video streaming, and evidence inspection.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import EVIDENCE_DIR, VIDEOS_DIR, BASE_DIR
from src.database import db
from src.detector import Detector
from src.tracker import ObjectTracker
from src.video_stream import VideoStreamManager
from src.policy import policy_manager

app = FastAPI(
    title="ExamGuard: AI-Based Exam Environment Monitoring System",
    description="Intelligent invigilator assistant for automated exam anomaly and object monitoring.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core AI Components
detector = Detector()
tracker = ObjectTracker(detector)
stream_manager = VideoStreamManager(detector, tracker)

# Request Models
class ReviewRequest(BaseModel):
    review_status: str  # 'FLAGGED_INFRACTION', 'DISMISSED_FALSE_ALARM', 'PENDING'
    invigilator_notes: str = ""
    reviewer: str = "Invigilator"

class ControlRequest(BaseModel):
    source: Optional[str] = None
    camera_id: Optional[str] = None
    confidence_threshold: Optional[float] = None
    enable_tracking: Optional[bool] = None

class PresetRequest(BaseModel):
    preset: str

class PolicyToggleRequest(BaseModel):
    item: str
    field: str  # 'allowed' or 'enabled'
    value: bool

# --- API ENDPOINTS ---

@app.get("/api/status")
def get_system_status():
    """System health, loaded model information, video stream status, and active exam policy."""
    policy_state = policy_manager.get_policy_state()
    return {
        "status": "ONLINE",
        "system": "ExamGuard AI Monitoring System",
        "active_model": detector.model_path if detector else "None",
        "is_custom_model": detector.is_custom_model if detector else False,
        "camera_active": stream_manager.is_running,
        "camera_id": stream_manager.camera_id,
        "video_source": str(stream_manager.source),
        "fps": round(stream_manager.current_fps, 1),
        "confidence_threshold": stream_manager.confidence_threshold,
        "tracking_enabled": stream_manager.is_tracking_enabled,
        "frames_processed": stream_manager.total_frames_processed,
        "active_preset": policy_manager.active_preset_id,
        "active_preset_name": policy_state["active_preset_name"],
        "active_preset_badge": policy_state["active_preset_badge"]
    }

@app.get("/api/policy")
def get_exam_policy():
    """Retrieve current exam hall item policy rules, items, and available presets."""
    return policy_manager.get_policy_state()

@app.post("/api/policy/preset")
def set_policy_preset(req: PresetRequest):
    """Switch active exam hall policy to a preset (Standard, Scientific, Open-Book, Lab)."""
    success = policy_manager.apply_preset(req.preset)
    if not success:
        raise HTTPException(status_code=400, detail=f"Unknown policy preset: '{req.preset}'")
    return {
        "status": "SUCCESS",
        "preset": req.preset,
        "policy": policy_manager.get_policy_state()
    }

@app.post("/api/policy/toggle")
def toggle_item_permission(req: PolicyToggleRequest):
    """Toggle whether a specific item is allowed vs. prohibited or enabled for detection."""
    if req.field == "allowed":
        success = policy_manager.set_item_rule(req.item, allowed=req.value)
    elif req.field == "enabled":
        success = policy_manager.set_item_rule(req.item, enabled=req.value)
    else:
        raise HTTPException(status_code=400, detail="Invalid field; must be 'allowed' or 'enabled'")
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Item '{req.item}' not found in policy registry")
    return {
        "status": "SUCCESS",
        "item": req.item,
        "field": req.field,
        "value": req.value,
        "policy": policy_manager.get_policy_state()
    }

@app.get("/api/stats")
def get_dashboard_stats():
    """Aggregated counters for Invigilator Dashboard KPI widgets."""
    return db.get_statistics()

@app.get("/api/alerts")
def get_alerts(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    severity: Optional[str] = None
):
    """Fetch paginated alert logs with optional filtering."""
    alerts = db.get_alerts(limit=limit, offset=offset, status=status, severity=severity)
    return {"alerts": alerts, "count": len(alerts)}

@app.post("/api/alerts/clear")
def clear_all_alerts():
    """Clear all stored alerts, reset total counts, and wipe cooldown trackers."""
    deleted_count = db.clear_all_alerts()
    from src.alerts import alert_manager
    alert_manager.cooldown_tracker.clear()
    alert_manager.persistence_tracker.clear()
    return {
        "status": "SUCCESS",
        "deleted_count": deleted_count,
        "message": f"Successfully cleared {deleted_count} alerts and reset KPI counters."
    }

@app.get("/api/alerts/{alert_id}")
def get_alert_detail(alert_id: str):
    """Retrieve detailed record for a specific alert."""
    record = db.get_alert_by_id(alert_id)
    if not record:
        raise HTTPException(status_code=404, detail="Alert record not found")
    return record

@app.post("/api/alerts/{alert_id}/review")
def review_alert(alert_id: str, req: ReviewRequest):
    """Human-in-the-Loop review endpoint: mark as verified infraction or dismiss as false alarm."""
    success = db.update_review(
        alert_id=alert_id,
        review_status=req.review_status,
        notes=req.invigilator_notes,
        reviewer=req.reviewer
    )
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or update failed")
    return {"status": "SUCCESS", "alert_id": alert_id, "review_status": req.review_status}

@app.get("/api/evidence/{filename}")
def get_evidence_image(filename: str):
    """Serves the visual snapshot captured at the moment of alert."""
    file_path = EVIDENCE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Evidence frame not found")
    return FileResponse(file_path, media_type="image/jpeg")

@app.get("/api/video_feed")
def get_video_feed():
    """Live MJPEG video stream with real-time detection & ByteTrack bounding boxes."""
    return StreamingResponse(
        stream_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/camera/control")
def control_camera(req: ControlRequest):
    """Configure active camera, change input video, or adjust detection sensitivity."""
    if req.source is not None:
        stream_manager.start_source(req.source)
    if req.camera_id is not None:
        stream_manager.camera_id = req.camera_id
    if req.confidence_threshold is not None:
        stream_manager.confidence_threshold = max(0.05, min(0.95, req.confidence_threshold))
    if req.enable_tracking is not None:
        stream_manager.is_tracking_enabled = req.enable_tracking
        
    return {
        "status": "UPDATED",
        "camera_id": stream_manager.camera_id,
        "source": str(stream_manager.source),
        "confidence_threshold": stream_manager.confidence_threshold,
        "tracking_enabled": stream_manager.is_tracking_enabled
    }

@app.get("/api/videos")
def list_available_videos():
    """List sample and uploaded exam videos."""
    videos = []
    for f in VIDEOS_DIR.glob("*.*"):
        if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            videos.append({"name": f.name, "path": str(f), "size_mb": round(f.stat().st_size / (1024*1024), 2)})
    return {"videos": videos}

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a test video file to monitor."""
    save_path = VIDEOS_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Automatically switch stream to the uploaded video
    stream_manager.start_source(str(save_path))
    
    return {
        "status": "SUCCESS",
        "filename": file.filename,
        "path": str(save_path),
        "message": "Video uploaded and stream initialized."
    }

# --- FRONTEND DASHBOARD ---

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the complete single-page interactive Invigilator Dashboard."""
    dashboard_path = BASE_DIR / "frontend" / "index.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ExamGuard Invigilator Dashboard file not found</h1>"
