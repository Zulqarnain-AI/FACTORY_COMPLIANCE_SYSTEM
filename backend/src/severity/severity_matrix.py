"""
severity_matrix.py — Module 2: Severity Categorization Matrix

Evaluates DetectionRecord objects and assigns/confirms a risk severity tier.
The base severity is pre-assigned by policy_parser.py from policy callout language.
This module refines severity based on contextual factors:
  - Personnel proximity (are people near the hazard?)
  - Concurrent violations (multiple simultaneous events escalate severity)
  - Frequency within clip (repeated violations in same clip = higher risk)

Policy-derived base severity (from Section 8 + callout analysis):
  Class 0 (Walkway Violation)        → HIGH   (WARNING callout, Section 3.3)
  Class 1 (Unauthorized Intervention)→ CRITICAL (CRITICAL SAFETY NOTICE, Section 4.3)
  Class 2 (Opened Panel Cover)       → HIGH   (WARNING callout, Section 5.2)
  Class 3 (Forklift Overload)        → CRITICAL (CRITICAL SAFETY NOTICE, Section 6.3)
"""

import logging
from collections import Counter
from dataclasses import dataclass

from src.detection.detection_engine import DetectionRecord

logger = logging.getLogger(__name__)

# Severity tier ordering (for comparison logic)
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class SeverityResult:
    event_id: str
    base_severity: str          # From policy callout
    final_severity: str         # After contextual refinement
    escalation_reason: str      # Why severity was changed (if at all)


class SeverityMatrix:
    """
    Module 2: Assigns final severity tiers to detected violations.

    The assignment of violations to tiers must be driven by the compliance policy content —
    specifically, the policy's descriptions of each behavior's hazard context, frequency data,
    and alerting language (per assignment spec).

    Tier definitions (from assignment spec):
    LOW:    State-based finding, no immediate personnel proximity
    MEDIUM: Behavioral deviation, personnel present but not in immediate danger
    HIGH:   Active unsafe behavior with concurrent personnel exposure
    CRITICAL: Immediate danger; highest-consequence hazard category in policy
    """

    # Violation frequency threshold: if same class_id fires >N times in one clip → escalate
    FREQUENCY_ESCALATION_THRESHOLD = 3

    def evaluate(
        self,
        records: list[DetectionRecord],
        concurrent_frame_window: int = 30  # Frames within which to consider violations "concurrent"
    ) -> list[DetectionRecord]:
        """
        Evaluates all violation records from a clip and applies contextual severity refinement.
        Returns the same list with updated severity values.
        """
        if not records:
            return records

        # Count violations per class in this clip (frequency signal)
        class_counts = Counter(r.class_id for r in records)

        # Build a frame → violations map for concurrency detection
        frame_violations: dict[int, list[DetectionRecord]] = {}
        for r in records:
            frame_violations.setdefault(r.frame_number, []).append(r)

        for record in records:
            base = record.severity
            final = base
            reason = "Severity assigned from policy callout language."

            # --- Rule 1: Frequency escalation ---
            # Policy Section 3.3 WARNING: "highest-frequency unsafe behavior"
            # If a behavior is detected repeatedly, it indicates systemic non-compliance
            if class_counts[record.class_id] > self.FREQUENCY_ESCALATION_THRESHOLD:
                candidate = self._escalate(final)
                if candidate != final:
                    final = candidate
                    reason = (
                        f"Escalated from {base} to {final}: violation class '{record.behavior_class}' "
                        f"detected {class_counts[record.class_id]} times in clip, indicating "
                        f"systemic non-compliance (threshold: {self.FREQUENCY_ESCALATION_THRESHOLD})."
                    )

            # --- Rule 2: Concurrent multi-class violations in same frame window ---
            # Multiple simultaneous violations indicate compound hazard exposure
            nearby_frames = self._get_nearby_violations(
                record.frame_number, frame_violations, concurrent_frame_window
            )
            concurrent_classes = set(v.class_id for v in nearby_frames if v.event_id != record.event_id)

            if len(concurrent_classes) >= 2:
                candidate = self._escalate(final)
                if candidate != final and candidate not in ("", None):
                    final = candidate
                    reason = (
                        f"Escalated from {base} to {final}: concurrent violations of classes "
                        f"{concurrent_classes} detected within {concurrent_frame_window} frames, "
                        f"indicating compound hazard exposure."
                    )

            # --- Rule 3: Class 2 (Panel) with nearby person = elevate LOW→MEDIUM, MEDIUM→HIGH ---
            # Policy: "condition is an unsafe behavior event regardless of how long the panel has been open
            # or whether personnel are in the immediate vicinity at the moment of detection"
            # BUT having personnel nearby makes it more urgent
            if record.class_id == 2:
                nearby_persons = [v for v in nearby_frames if v.class_id == 0]  # Walkway violations = persons present
                if nearby_persons and SEVERITY_ORDER[final] < SEVERITY_ORDER["HIGH"]:
                    final = "HIGH"
                    reason = f"Panel violation (Class 2) upgraded to HIGH: personnel detected in proximity."

            record.severity = final
            if final != base:
                logger.debug(f"[{record.event_id[:8]}] {base} → {final}: {reason}")

        return records

    def _escalate(self, current: str) -> str:
        """Move up one severity tier (max: CRITICAL)."""
        idx = SEVERITY_ORDER.get(current, 0)
        return SEVERITY_LABELS[min(idx + 1, 3)]

    def _get_nearby_violations(
        self, frame_number: int,
        frame_violations: dict[int, list[DetectionRecord]],
        window: int
    ) -> list[DetectionRecord]:
        """Returns all violations within ±window frames of the given frame number."""
        nearby = []
        for fn, violations in frame_violations.items():
            if abs(fn - frame_number) <= window:
                nearby.extend(violations)
        return nearby


severity_matrix = SeverityMatrix()
