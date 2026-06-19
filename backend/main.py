"""
main.py — FastAPI Application Entry Point

API Endpoints:
  GET  /                        → Health check
  GET  /api/policy/rules        → Return all parsed compliance rules
  POST /api/process             → Upload & process a video clip (full pipeline)
  GET  /api/reports             → List compliance reports (with filters)
  GET  /api/reports/{event_id}  → Get single report
  GET  /api/export/csv          → Download full log as CSV
  GET  /api/export/json         → Download full log as JSON
  GET  /api/stats               → Dashboard summary statistics
  WS   /ws/alerts               → WebSocket for real-time HIGH/CRIT alerts

Processing Pipeline per /api/process call:
  1. Save uploaded video to temp dir
  2. Run DetectionEngine → List[DetectionRecord]
  3. Run SeverityMatrix → refine severities
  4. Run EscalationPipeline → route events + broadcast WebSocket alerts
  5. Run ReportGenerator → write to MongoDB
  6. Return summary response
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import settings
from src.policy.policy_parser import policy_parser
from src.detection.detection_engine import detection_engine
from src.severity.severity_matrix import severity_matrix
from src.escalation.escalation_pipeline import escalation_pipeline, ws_manager
from src.reports.report_generator import report_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MongoDB client lifecycle
# ---------------------------------------------------------------------------

mongo_client: AsyncIOMotorClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MongoDB and warm up policy parser on startup."""
    global mongo_client
    logger.info("🏭 Factory Compliance System starting up...")

    # Connect MongoDB
    mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db = mongo_client[settings.MONGODB_DB]
    logger.info(f"MongoDB connected: {settings.MONGODB_URI}/{settings.MONGODB_DB}")

    # Warm up policy parser (triggers Groq LLM call on first access)
    rules = policy_parser.get_rules()
    logger.info(f"Policy loaded: {len(rules)} compliance rules active.")

    # Ensure output directory exists
    Path(settings.REPORTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    if mongo_client:
        mongo_client.close()
    logger.info("Factory Compliance System shut down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Factory Compliance & Alert Escalation System",
    description="KMP-OHS-POL-001 — Real-time OHS violation detection and reporting",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> AsyncIOMotorDatabase:
    return app.state.db


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {
        "status": "online",
        "system": "Factory Compliance & Alert Escalation System",
        "policy": "KMP-OHS-POL-001",
        "active_rules": len(policy_parser.get_rules()),
        "ws_connections": len(ws_manager.active_connections),
    }


# ---------------------------------------------------------------------------
# Policy Routes
# ---------------------------------------------------------------------------

@app.get("/api/policy/rules")
async def get_policy_rules():
    """Returns all parsed compliance rules (for dashboard reference)."""
    return {
        "rules": policy_parser.get_rules_as_dict(),
        "total": len(policy_parser.get_rules())
    }


# ---------------------------------------------------------------------------
# Video Processing — Core Pipeline
# ---------------------------------------------------------------------------

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """
    Main endpoint: accepts a video clip, runs full compliance detection pipeline,
    returns violation summary.

    Pipeline:
    1. Save file to temp location
    2. Detection Engine (YOLO + OpenCV)
    3. Severity Matrix (contextual refinement)
    4. Escalation Pipeline (WebSocket alerts for HIGH/CRIT)
    5. Report Generation (MongoDB write + LLM description enrichment)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Processing clip: {file.filename} ({len(content) / 1024:.1f} KB)")

        # --- Module 1: Detection ---
        raw_records = detection_engine.process_clip(tmp_path)

        # --- Module 2: Severity refinement ---
        records = severity_matrix.evaluate(raw_records)

        # --- Module 3: Escalation (WebSocket alerts) ---
        escalated = await escalation_pipeline.process(records)

        # --- Module 4: Report generation ---
        reports = await report_generator.write_reports(db, escalated)

        # Build response summary
        severity_counts = {}
        for r in records:
            severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1

        return {
            "status": "processed",
            "clip_id": file.filename,
            "total_violations": len(records),
            "severity_breakdown": severity_counts,
            "reports_written": len(reports),
            "violations": [
                {
                    "event_id": r.event_id,
                    "timestamp": r.timestamp,
                    "behavior_class": r.behavior_class,
                    "severity": r.severity,
                    "policy_rule_ref": r.policy_rule_ref,
                    "zone": r.zone,
                    "confidence": r.confidence,
                    "frame_number": r.frame_number,
                    "event_description": r.event_description,
                    "frame_snapshot_b64": r.frame_snapshot_b64,
                }
                for r in records
            ]
        }

    finally:
        os.unlink(tmp_path)  # Clean up temp file


# ---------------------------------------------------------------------------
# Reports / Historical Log
# ---------------------------------------------------------------------------

@app.get("/api/reports")
async def list_reports(
    severity: str | None = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    behavior_class: str | None = Query(None, description="Filter by behavior class (partial match)"),
    date_from: str | None = Query(None, description="ISO 8601 start timestamp"),
    date_to: str | None = Query(None, description="ISO 8601 end timestamp"),
    limit: int = Query(200, le=1000),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """
    Returns filtered compliance reports from MongoDB.
    Powers the dashboard's Historical Log & Export view (Module 5, View C).
    """
    reports = await report_generator.get_all_reports(
        db, severity=severity, behavior_class=behavior_class,
        date_from=date_from, date_to=date_to, limit=limit
    )
    return {"reports": reports, "count": len(reports)}


@app.get("/api/reports/{event_id}")
async def get_report(event_id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    """Returns a single compliance report by event_id."""
    collection = db[settings.REPORTS_COLLECTION]
    doc = await collection.find_one({"event_id": event_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Report not found: {event_id}")
    doc.pop("_id", None)
    return doc


# ---------------------------------------------------------------------------
# Export Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/export/csv")
async def export_csv(
    severity: str | None = Query(None),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Downloads full (or filtered) compliance log as CSV."""
    reports = await report_generator.get_all_reports(db, severity=severity, limit=10000)
    csv_content = report_generator.export_to_csv(reports)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=compliance_log.csv"}
    )


@app.get("/api/export/json")
async def export_json_file(
    severity: str | None = Query(None),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Downloads full (or filtered) compliance log as JSON."""
    reports = await report_generator.get_all_reports(db, severity=severity, limit=10000)
    json_content = report_generator.export_to_json(reports)
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=compliance_log.json"}
    )


# ---------------------------------------------------------------------------
# Dashboard Statistics
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def get_stats(db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    Returns summary statistics for the dashboard overview panel.
    """
    collection = db[settings.REPORTS_COLLECTION]
    total = await collection.count_documents({})

    severity_counts = {}
    for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        severity_counts[sev] = await collection.count_documents({"severity": sev})

    class_counts = {}
    for cls in ["Safe Walkway Violation", "Unauthorized Intervention",
                "Opened Panel Cover", "Carrying Overload with Forklift"]:
        class_counts[cls] = await collection.count_documents({"behavior_class": cls})

    # Most recent 5 events
    recent_cursor = collection.find(
        {}, {"event_id": 1, "timestamp": 1, "behavior_class": 1, "severity": 1, "zone": 1}
    ).sort("timestamp", -1).limit(5)

    recent = []
    async for doc in recent_cursor:
        doc.pop("_id", None)
        recent.append(doc)

    return {
        "total_violations": total,
        "by_severity": severity_counts,
        "by_class": class_counts,
        "recent_events": recent,
        "active_ws_connections": len(ws_manager.active_connections),
    }


# ---------------------------------------------------------------------------
# WebSocket — Real-time Alert Stream (Module 3 output)
# ---------------------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for the dashboard's real-time alert stream.
    HIGH and CRITICAL violations are pushed here immediately by the escalation pipeline.

    Message format:
    {
      "type": "VIOLATION_ALERT",
      "severity": "CRITICAL",
      "event_id": "...",
      "timestamp": "...",
      "behavior_class": "...",
      "policy_rule_ref": "...",
      "event_description": "...",
      "zone": "Zone-1",
      "clip_id": "video_clip.mp4",
      "escalation_action": "Real-time alert triggered + DB log",
      "confidence": 0.87,
      "frame_snapshot_b64": "..."   // base64 JPEG, may be null
    }
    """
    await ws_manager.connect(websocket)
    # Send welcome/connection confirmation
    await ws_manager.send_personal(websocket, {
        "type": "CONNECTED",
        "message": "Factory Compliance Alert Stream active. Monitoring for HIGH and CRITICAL violations.",
        "policy": "KMP-OHS-POL-001"
    })

    try:
        while True:
            # Keep the connection alive; alerts are pushed by escalation_pipeline.process()
            data = await websocket.receive_text()
            # Handle ping-pong from client
            if data == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("Dashboard client disconnected from alert stream.")


# ---------------------------------------------------------------------------
# Application Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Factory Compliance System server on http://0.0.0.0:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
