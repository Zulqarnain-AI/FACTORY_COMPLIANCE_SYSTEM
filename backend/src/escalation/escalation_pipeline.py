"""
escalation_pipeline.py — Module 3: Escalation Pipeline

Routes violation events to the correct downstream channel based on severity tier:

  LOW / MEDIUM   → Persistent database log only (no real-time alert)
  HIGH / CRITICAL → Real-time WebSocket alert to dashboard + persistent database log

WebSocket Manager:
  - Maintains a set of active WebSocket connections (from the dashboard frontend).
  - Broadcasts HIGH/CRITICAL violation payloads to ALL connected clients immediately.
  - Connection lifecycle is managed here (connect / disconnect).
"""

import json
import logging
from enum import Enum
from fastapi import WebSocket

from src.detection.detection_engine import DetectionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation tier routing
# ---------------------------------------------------------------------------

class EscalationAction(str, Enum):
    LOG_ONLY = "Logged to DB"
    ALERT_AND_LOG = "Real-time alert triggered + DB log"


ESCALATION_ROUTING = {
    "LOW": EscalationAction.LOG_ONLY,
    "MEDIUM": EscalationAction.LOG_ONLY,
    "HIGH": EscalationAction.ALERT_AND_LOG,
    "CRITICAL": EscalationAction.ALERT_AND_LOG,
}

ALERT_SEVERITIES = {"HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages WebSocket connections from the React dashboard.
    All HIGH/CRITICAL violations are broadcast to every connected client.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, payload: dict):
        """Send a JSON payload to ALL connected dashboard clients."""
        message = json.dumps(payload)
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.append(ws)
        # Clean up dead connections
        for ws in disconnected:
            self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, payload: dict):
        """Send to a specific connection (e.g., on initial connect)."""
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Personal send failed: {e}")


# Global singleton connection manager (shared across FastAPI routes)
ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Escalation Pipeline
# ---------------------------------------------------------------------------

class EscalationPipeline:
    """
    Processes a list of violations: determines escalation action per severity,
    triggers WebSocket alerts for HIGH/CRITICAL events.
    """

    def __init__(self, manager: ConnectionManager):
        self.manager = manager

    def get_action(self, severity: str) -> EscalationAction:
        """Look up the required escalation action for a severity tier."""
        return ESCALATION_ROUTING.get(severity, EscalationAction.LOG_ONLY)

    def requires_alert(self, severity: str) -> bool:
        return severity in ALERT_SEVERITIES

    async def process(self, records: list[DetectionRecord]) -> list[tuple[DetectionRecord, EscalationAction]]:
        """
        For each violation record:
        1. Determine escalation action.
        2. For HIGH/CRITICAL: broadcast WebSocket alert to dashboard.
        3. Return list of (record, action) for the report writer.
        """
        results = []

        for record in records:
            action = self.get_action(record.severity)

            if self.requires_alert(record.severity):
                alert_payload = self._build_alert_payload(record, action)
                await self.manager.broadcast(alert_payload)
                logger.info(
                    f"🚨 [{record.severity}] Alert broadcast: {record.behavior_class} "
                    f"in {record.zone} (clip: {record.clip_id})"
                )
            else:
                logger.debug(f"[{record.severity}] Logged: {record.behavior_class}")

            results.append((record, action))

        return results

    def _build_alert_payload(self, record: DetectionRecord, action: EscalationAction) -> dict:
        """Build the WebSocket message payload for HIGH/CRITICAL alerts."""
        return {
            "type": "VIOLATION_ALERT",
            "severity": record.severity,
            "event_id": record.event_id,
            "timestamp": record.timestamp,
            "clip_id": record.clip_id,
            "zone": record.zone,
            "behavior_class": record.behavior_class,
            "policy_rule_ref": record.policy_rule_ref,
            "event_description": record.event_description,
            "escalation_action": action.value,
            "confidence": record.confidence,
            "frame_number": record.frame_number,
            # Include thumbnail if available (base64 JPEG)
            "frame_snapshot_b64": record.frame_snapshot_b64,
        }


# Global singleton
escalation_pipeline = EscalationPipeline(ws_manager)
