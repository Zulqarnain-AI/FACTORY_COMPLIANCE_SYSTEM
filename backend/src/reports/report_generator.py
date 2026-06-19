"""
report_generator.py — Module 4: Automated Report Generation

Produces structured, immutable compliance records for every detected violation.
Stores records in MongoDB and exports to JSON/CSV on demand.

Required fields per assignment spec:
  event_id, timestamp, clip_id, zone, behavior_class, policy_rule_ref,
  event_description, severity, escalation_action

The event_description is enriched by Groq LLM for human-readable reporting.
"""

import csv
import json
import logging
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from groq import Groq
from motor.motor_asyncio import AsyncIOMotorDatabase

from config import settings
from src.detection.detection_engine import DetectionRecord
from src.escalation.escalation_pipeline import EscalationAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MongoDB document schema (mirrors required report fields)
# ---------------------------------------------------------------------------

def build_report_document(
    record: DetectionRecord,
    action: EscalationAction,
    enriched_description: str | None = None
) -> dict:
    """
    Builds a MongoDB document from a DetectionRecord.
    This is the canonical compliance report format.
    """
    return {
        "_id": record.event_id,               # Use event_id as MongoDB _id for idempotency
        "event_id": record.event_id,
        "timestamp": record.timestamp,
        "clip_id": record.clip_id,
        "zone": record.zone,
        "behavior_class": record.behavior_class,
        "class_id": record.class_id,
        "policy_rule_ref": record.policy_rule_ref,
        "event_description": enriched_description or record.event_description,
        "severity": record.severity,
        "escalation_action": action.value,
        "confidence": record.confidence,
        "frame_number": record.frame_number,
        "bbox": record.bbox,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Frame snapshot stored separately (large field) — omit from list views
        "has_snapshot": record.frame_snapshot_b64 is not None,
    }


# ---------------------------------------------------------------------------
# LLM Description Enricher
# ---------------------------------------------------------------------------

class DescriptionEnricher:
    """
    Uses Groq LLM to rewrite technical detection descriptions into
    clear, professional compliance report language.
    Called once per violation event.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def enrich(self, record: DetectionRecord) -> str:
        """
        Returns an enriched human-readable event description.
        Falls back to the original description on any LLM error.
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a compliance officer writing official safety violation reports "
                            "for a manufacturing facility. Rewrite the following technical detection "
                            "note into a clear, professional 2-3 sentence compliance report entry. "
                            "Be precise about what was observed. Do not add information not in the input. "
                            "Output only the rewritten description text, no preamble."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Violation type: {record.behavior_class}\n"
                            f"Policy reference: {record.policy_rule_ref}\n"
                            f"Severity: {record.severity}\n"
                            f"Original detection note: {record.event_description}"
                        )
                    }
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM description enrichment failed for {record.event_id[:8]}: {e}")
            return record.event_description


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Module 4: Writes compliance reports to MongoDB and optionally to files.
    """

    def __init__(self):
        self.enricher = DescriptionEnricher()
        self.output_dir = Path(settings.REPORTS_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def write_reports(
        self,
        db: AsyncIOMotorDatabase,
        escalated: list[tuple[DetectionRecord, EscalationAction]]
    ) -> list[dict]:
        """
        Writes compliance reports to MongoDB for all violations.
        Returns the list of report documents written.
        """
        collection = db[settings.REPORTS_COLLECTION]
        documents = []

        for record, action in escalated:
            # Enrich description with LLM
            enriched_desc = self.enricher.enrich(record)

            # Build report document
            doc = build_report_document(record, action, enriched_desc)
            documents.append(doc)

            # Upsert to MongoDB (idempotent on event_id)
            try:
                await collection.replace_one(
                    {"_id": record.event_id},
                    doc,
                    upsert=True
                )
                logger.debug(f"Report written: {record.event_id[:8]} [{record.severity}]")
            except Exception as e:
                logger.error(f"Failed to write report {record.event_id[:8]}: {e}")

        logger.info(f"Module 4: {len(documents)} compliance reports written to MongoDB.")
        return documents

    async def get_all_reports(
        self,
        db: AsyncIOMotorDatabase,
        severity: str | None = None,
        behavior_class: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Retrieves compliance reports from MongoDB with optional filters.
        Used by the dashboard's Historical Log (View C).
        """
        collection = db[settings.REPORTS_COLLECTION]
        query: dict = {}

        if severity:
            query["severity"] = severity.upper()
        if behavior_class:
            query["behavior_class"] = {"$regex": behavior_class, "$options": "i"}
        if date_from:
            query.setdefault("timestamp", {})["$gte"] = date_from
        if date_to:
            query.setdefault("timestamp", {})["$lte"] = date_to

        cursor = collection.find(
            query,
            {"has_snapshot": 0, "bbox": 0}   # Exclude heavy fields from list view
        ).sort("timestamp", -1).limit(limit)

        reports = []
        async for doc in cursor:
            doc.pop("_id", None)   # Remove MongoDB internal _id for JSON serialization
            reports.append(doc)

        return reports

    def export_to_csv(self, reports: list[dict]) -> str:
        """
        Exports a list of report dicts to CSV format (string).
        Used by the dashboard's Export button.
        """
        if not reports:
            return ""

        fields = [
            "event_id", "timestamp", "clip_id", "zone", "behavior_class",
            "policy_rule_ref", "event_description", "severity",
            "escalation_action", "confidence", "frame_number", "created_at"
        ]

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
        return output.getvalue()

    def export_to_json(self, reports: list[dict]) -> str:
        """Exports reports to a JSON string."""
        return json.dumps(reports, indent=2, default=str)


# Global singleton
report_generator = ReportGenerator()
