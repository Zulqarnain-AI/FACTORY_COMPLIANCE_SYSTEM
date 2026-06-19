"""
detection_engine.py — Module 1: Vision Detection Engine

Processes video clips frame-by-frame using YOLOv8 + OpenCV to detect the four
behavioral violation classes defined in the OHS Compliance Policy Manual.

Detection Logic (grounded in policy observable indicators):
─────────────────────────────────────────────────────────────
Class 0 (Walkway Violation, Section 3.3.2):
  - Detect 'person' bounding boxes.
  - Check if person centroid falls OUTSIDE the green walkway zone polygon.
  - Zone is defined by config fractions (calibrated per camera).

Class 1 (Unauthorized Intervention, Section 4.3.2):
  - Detect 'person' near 'machine/equipment' (proximity heuristic).
  - Check vest color: green vest = authorized, red/dark = unauthorized.
  - Uses HSV color analysis on the torso crop of each detected person.

Class 2 (Opened Panel Cover, Section 5.2.2):
  - Detect electrical panel objects.
  - Check if panel cover state is open (detected as 'open_panel' class).

Class 3 (Forklift Overload, Section 6.3.2):
  - Detect forklifts and count associated block objects on forks.
  - If block count >= 3 → violation.

Model Strategy:
  - We use YOLOv8 with COCO pretrained weights as a base.
  - COCO 'person' class (0) is used directly.
  - Custom classes (vest colors, open panels, blocks) require a fine-tuned model.
  - A FALLBACK_SIMULATION mode is provided so the pipeline runs end-to-end
    without a fine-tuned model, using heuristics + color detection.
"""

import cv2
import numpy as np
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Suppress a noisy FutureWarning emitted from ultralytics/torch when
# calling `torch.load(..., weights_only=False)` (library default).
# This avoids alarming the user while the underlying library remains
# functional. If you prefer to address it at the source, upgrade
# ultralytics or change the call to `weights_only=True` when supported.
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.load.*weights_only=False.*",
)

from ultralytics import YOLO

from config import settings
from src.policy.policy_parser import policy_parser, ComplianceRule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOLO class name constants
# Update these to match your fine-tuned model's class names exactly.
# ---------------------------------------------------------------------------

# Standard COCO classes we use from the base model
COCO_PERSON = "person"
COCO_FORKLIFT_PROXIES = {"truck", "car", "vehicle"}  # Proxy until fine-tuned

# Custom class names (your fine-tuned model should output these)
CUSTOM_GREEN_VEST = "green_vest"
CUSTOM_RED_BLACK_VEST = "red_black_vest"
CUSTOM_OPEN_PANEL = "open_panel"
CUSTOM_CLOSED_PANEL = "closed_panel"
CUSTOM_BLOCK = "block"
CUSTOM_FORKLIFT = "forklift"


# ---------------------------------------------------------------------------
# Detection result data model
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """Structured output for a single detected violation (feeds Module 2+)."""
    event_id: str
    timestamp: str                  # ISO 8601 wall-clock time
    clip_id: str                    # Source video filename
    zone: str                       # Inferred zone label
    class_id: int                   # 0-3 per policy Section 8
    behavior_class: str             # e.g. "Safe Walkway Violation"
    policy_rule_ref: str            # e.g. "Section 3.3.2"
    event_description: str          # Human-readable (filled by LLM in reports module)
    severity: str                   # Pre-filled from policy (final value from Module 2)
    frame_number: int
    confidence: float
    bbox: list[int]                 # [x1, y1, x2, y2] in pixels
    frame_snapshot_b64: Optional[str] = None   # Base64 JPEG of annotated frame


# ---------------------------------------------------------------------------
# Walkway Zone Helper
# ---------------------------------------------------------------------------

class WalkwayZone:
    """
    Represents the Designated Safe Walkway (green floor markings, Section 3.2).

    In production: calibrate the polygon points by inspecting actual camera frames
    and marking the green walkway boundaries as (x, y) pixel coordinates.
    Here we use configurable fractional bounds as a starting approximation.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Convert fractional config to pixel coordinates
        x_min = int(settings.WALKWAY_X_MIN_FRAC * frame_w)
        x_max = int(settings.WALKWAY_X_MAX_FRAC * frame_w)
        y_min = int(settings.WALKWAY_Y_MIN_FRAC * frame_h)
        y_max = int(settings.WALKWAY_Y_MAX_FRAC * frame_h)

        # Rectangle polygon (replace with actual green-marking polygon for deployment)
        self.polygon = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.int32)

    def is_outside(self, cx: int, cy: int) -> bool:
        """
        Returns True if the point (cx, cy) is OUTSIDE the walkway zone.
        Uses OpenCV's pointPolygonTest (positive = inside, negative = outside).
        """
        result = cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False)
        return result < 0  # Negative value = outside polygon

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw walkway boundary on frame for visualization."""
        overlay = frame.copy()
        cv2.polylines(overlay, [self.polygon], isClosed=True, color=(0, 200, 0), thickness=2)
        cv2.fillPoly(overlay, [self.polygon], color=(0, 200, 0))
        return cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


# ---------------------------------------------------------------------------
# Vest Color Analyzer (HSV-based — Policy Section 4.2)
# ---------------------------------------------------------------------------

class VestColorAnalyzer:
    """
    Classifies the vest color of a person cropped from a frame.

    Policy Section 4.2 defines two vest types:
    - Green vest  → Authorized (safe for equipment interaction)
    - Red-black vest → General personnel (NOT authorized for equipment intervention)

    Method: Analyze the HSV histogram of the person's torso region (upper 60% of bbox).
    """

    # HSV ranges for green (OpenCV H range: 0-179)
    GREEN_H_MIN, GREEN_H_MAX = 35, 85
    GREEN_S_MIN, GREEN_S_MAX = 80, 255
    GREEN_V_MIN, GREEN_V_MAX = 60, 255

    # HSV ranges for red (wraps around 0/179 in OpenCV)
    RED_H_RANGES = [(0, 15), (160, 179)]
    RED_S_MIN, RED_S_MAX = 100, 255
    RED_V_MIN, RED_V_MAX = 50, 255

    # Minimum green pixel fraction to classify as "green vest"
    GREEN_THRESHOLD = 0.08
    RED_THRESHOLD = 0.06

    def analyze(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
        """
        Analyzes vest color in the torso region of a detected person.

        Returns:
            "green"     → Authorized (green vest detected)
            "red_black" → Unauthorized (red/dark vest detected)
            "unknown"   → Cannot determine
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1

        # Torso = upper 60% of the bounding box (avoids legs / floor color)
        torso_y2 = y1 + int(h * 0.6)
        torso_crop = frame[y1:torso_y2, x1:x2]

        if torso_crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return "unknown"

        # --- Green detection ---
        green_mask = cv2.inRange(
            hsv,
            np.array([self.GREEN_H_MIN, self.GREEN_S_MIN, self.GREEN_V_MIN]),
            np.array([self.GREEN_H_MAX, self.GREEN_S_MAX, self.GREEN_V_MAX])
        )
        green_frac = np.count_nonzero(green_mask) / total_pixels

        # --- Red detection (two ranges due to HSV wraparound) ---
        red_mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.RED_H_RANGES:
            red_mask_total |= cv2.inRange(
                hsv,
                np.array([h_min, self.RED_S_MIN, self.RED_V_MIN]),
                np.array([h_max, self.RED_S_MAX, self.RED_V_MAX])
            ) # type: ignore
        red_frac = np.count_nonzero(red_mask_total) / total_pixels

        if green_frac >= self.GREEN_THRESHOLD:
            return "green"
        elif red_frac >= self.RED_THRESHOLD:
            return "red_black"
        else:
            return "unknown"  # Default: no clear vest → treat as unauthorized in equipment context


# ---------------------------------------------------------------------------
# Main Detection Engine
# ---------------------------------------------------------------------------

class DetectionEngine:
    """
    Processes video clips and returns a list of DetectionRecord objects
    representing all compliance violations found in the clip.
    """

    def __init__(self):
        logger.info(f"Loading YOLO model: {settings.YOLO_MODEL_PATH}")
        self.model = YOLO(settings.YOLO_MODEL_PATH)
        self.conf_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self.frame_interval = settings.FRAME_SAMPLE_INTERVAL
        self.vest_analyzer = VestColorAnalyzer()
        self._rules = policy_parser.get_rules()
        self._rule_map = {r.class_id: r for r in self._rules}
        self._model_labels = {
            str(name).lower()
            for name in getattr(self.model, "names", {}).values()
        }
        self._has_custom_compliance_classes = any(
            label in self._model_labels
            for label in {CUSTOM_GREEN_VEST, CUSTOM_RED_BLACK_VEST, CUSTOM_OPEN_PANEL, CUSTOM_BLOCK, CUSTOM_FORKLIFT}
        )
        if not self._has_custom_compliance_classes:
            logger.warning(
                "YOLO model '%s' does not expose custom compliance classes; "
                "forklift and block detection will use fallback heuristics.",
                settings.YOLO_MODEL_PATH,
            )

    def process_clip(self, video_path: str | Path) -> list[DetectionRecord]:
        """
        Entry point: processes a single video clip and returns all violations.

        Args:
            video_path: Path to the .mp4 (or other format) video clip.

        Returns:
            List of DetectionRecord, one per detected violation event.
        """
        video_path = Path(video_path)
        clip_id = video_path.name
        violations: list[DetectionRecord] = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return violations

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        walkway_zone = WalkwayZone(frame_w, frame_h)
        frame_number = 0
        now = datetime.now(timezone.utc)

        logger.info(f"Processing clip: {clip_id} ({frame_w}x{frame_h} @ {fps:.1f}fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Sample every Nth frame for performance
            if frame_number % self.frame_interval != 0:
                continue

            timestamp_sec = frame_number / fps
            timestamp_iso = (
                now.replace(
                    second=int(timestamp_sec) % 60,
                    microsecond=0
                ).isoformat()
            )

            # Run YOLO inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = results[0]

            # Parse detection boxes
            boxes = self._parse_boxes(detections, frame_w, frame_h)

            # Annotate frame with walkway zone
            annotated = walkway_zone.draw(frame.copy())

            # --- Check each violation class ---
            frame_violations = []

            frame_violations += self._check_walkway_violations(
                boxes, walkway_zone, frame_number, timestamp_iso, clip_id
            )
            frame_violations += self._check_intervention_violations(
                frame, boxes, frame_number, timestamp_iso, clip_id
            )
            frame_violations += self._check_panel_violations(
                boxes, frame_number, timestamp_iso, clip_id
            )
            frame_violations += self._check_forklift_violations(
                frame, boxes, frame_number, timestamp_iso, clip_id
            )

            # Annotate and attach frame snapshot to each violation
            for v in frame_violations:
                annotated_with_bbox = self._draw_violation(annotated.copy(), v)
                v.frame_snapshot_b64 = self._frame_to_b64(annotated_with_bbox)
                violations.append(v)

        cap.release()
        logger.info(f"Clip '{clip_id}': {len(violations)} violations detected.")
        return violations

    # -----------------------------------------------------------------------
    # Internal: Parse YOLO boxes into a structured dict
    # -----------------------------------------------------------------------

    def _parse_boxes(self, detections, frame_w: int, frame_h: int) -> dict[str, list]:
        """
        Parses YOLO result boxes into per-class lists.
        Keys: 'person', 'green_vest', 'red_black_vest', 'open_panel',
              'closed_panel', 'block', 'forklift'
        Each item: {'bbox': [x1,y1,x2,y2], 'conf': float, 'label': str}
        """
        parsed: dict[str, list] = {
            "person": [], "green_vest": [], "red_black_vest": [],
            "open_panel": [], "closed_panel": [], "block": [], "forklift": []
        }

        if detections.boxes is None:
            return parsed

        names = detections.names  # id → class name from the model

        for box in detections.boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            # Clip to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            item = {"bbox": [x1, y1, x2, y2], "conf": conf, "label": label}

            # Map COCO 'person' (class 0) directly
            if label == "person":
                parsed["person"].append(item)
            # Custom fine-tuned class names
            elif label in (CUSTOM_GREEN_VEST, "green-vest", "greenvest"):
                parsed["green_vest"].append(item)
            elif label in (CUSTOM_RED_BLACK_VEST, "red_black_vest", "red-vest"):
                parsed["red_black_vest"].append(item)
            elif label in (CUSTOM_OPEN_PANEL, "open-panel", "open_panel"):
                parsed["open_panel"].append(item)
            elif label in (CUSTOM_CLOSED_PANEL, "closed_panel", "closed-panel"):
                parsed["closed_panel"].append(item)
            elif label in (CUSTOM_BLOCK, "block", "pallet", "box"):
                parsed["block"].append(item)
            elif label in (CUSTOM_FORKLIFT, "forklift", "truck"):
                parsed["forklift"].append(item)

        return parsed

    # -----------------------------------------------------------------------
    # Class 0: Walkway Violation (Section 3.3.2)
    # -----------------------------------------------------------------------

    def _check_walkway_violations(
        self, boxes: dict, walkway_zone: WalkwayZone,
        frame_number: int, timestamp: str, clip_id: str
    ) -> list[DetectionRecord]:
        """
        Detects persons whose centroid is OUTSIDE the green walkway boundaries.
        Uses the WalkwayZone.is_outside() method (OpenCV polygon test).
        """
        violations = []
        rule = self._rule_map[0]

        for person in boxes["person"]:
            x1, y1, x2, y2 = person["bbox"]
            # Use foot position (bottom center) as the position reference
            cx = (x1 + x2) // 2
            cy = y2  # Bottom of bounding box = approximate foot position

            if walkway_zone.is_outside(cx, cy):
                violations.append(self._make_record(
                    rule=rule,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    clip_id=clip_id,
                    bbox=person["bbox"],
                    confidence=person["conf"],
                    description=(
                        f"Person detected at position ({cx}, {cy}) which is outside the "
                        f"green-marked Designated Safe Walkway boundary (Section 3.3.2). "
                        f"Foot position places individual in a machinery/forklift operation zone."
                    )
                ))

        return violations

    # -----------------------------------------------------------------------
    # Class 1: Unauthorized Intervention (Section 4.3.2)
    # -----------------------------------------------------------------------

    def _check_intervention_violations(
        self, frame: np.ndarray, boxes: dict,
        frame_number: int, timestamp: str, clip_id: str
    ) -> list[DetectionRecord]:
        """
        Detects persons interacting with equipment without the green authorization vest.

        Proximity heuristic: if a person bbox overlaps or is within 80px of a
        forklift/panel bbox, they are considered to be "interacting with equipment."
        Vest color is then analyzed via HSV color analysis on the torso region.
        """
        violations = []
        rule = self._rule_map[1]

        # Equipment proxies: panels + forklifts
        equipment_bboxes = (
            [d["bbox"] for d in boxes["open_panel"]] +
            [d["bbox"] for d in boxes["closed_panel"]] +
            [d["bbox"] for d in boxes["forklift"]]
        )

        for person in boxes["person"]:
            if not self._is_near_equipment(person["bbox"], equipment_bboxes, threshold_px=100):
                continue  # Not interacting with equipment → skip

            # Analyze vest color from HSV torso crop
            vest_color = self.vest_analyzer.analyze(frame, tuple(person["bbox"]))

            if vest_color != "green":
                vest_desc = "red-black vest" if vest_color == "red_black" else "no identifiable green vest"
                violations.append(self._make_record(
                    rule=rule,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    clip_id=clip_id,
                    bbox=person["bbox"],
                    confidence=person["conf"],
                    description=(
                        f"Person detected interacting with production equipment while wearing {vest_desc}. "
                        f"Green authorization vest not present. Per Section 4.3.2, this constitutes "
                        f"an Unauthorized Intervention regardless of the individual's stated role."
                    )
                ))

        return violations

    def _is_near_equipment(
        self, person_bbox: list[int], equipment_bboxes: list[list[int]], threshold_px: int
    ) -> bool:
        """Returns True if the person bbox is within threshold_px of any equipment bbox."""
        px1, py1, px2, py2 = person_bbox
        p_cx, p_cy = (px1 + px2) // 2, (py1 + py2) // 2

        for ex1, ey1, ex2, ey2 in equipment_bboxes:
            e_cx, e_cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
            dist = np.sqrt((p_cx - e_cx) ** 2 + (p_cy - e_cy) ** 2)
            if dist < threshold_px:
                return True
            # Also check direct bbox overlap
            if px1 < ex2 and px2 > ex1 and py1 < ey2 and py2 > ey1:
                return True
        return False

    # -----------------------------------------------------------------------
    # Class 2: Opened Panel Cover (Section 5.2.2)
    # -----------------------------------------------------------------------

    def _check_panel_violations(
        self, boxes: dict,
        frame_number: int, timestamp: str, clip_id: str
    ) -> list[DetectionRecord]:
        """
        Detects electrical panels in the open-cover state.
        The mere presence of an 'open_panel' detection = violation (Section 5.2.2).
        """
        violations = []
        rule = self._rule_map[2]

        for panel in boxes["open_panel"]:
            x1, y1, x2, y2 = panel["bbox"]
            violations.append(self._make_record(
                rule=rule,
                frame_number=frame_number,
                timestamp=timestamp,
                clip_id=clip_id,
                bbox=panel["bbox"],
                confidence=panel["conf"],
                description=(
                    f"Electrical panel cover observed in OPEN state at frame region "
                    f"({x1},{y1})-({x2},{y2}). Per Section 5.2.2, this condition is classified "
                    f"as unsafe regardless of duration or whether personnel are in the immediate "
                    f"vicinity. Panel must be closed immediately."
                )
            ))

        return violations

    # -----------------------------------------------------------------------
    # Class 3: Forklift Overload (Section 6.3.2)
    # -----------------------------------------------------------------------

    def _check_forklift_violations(
        self, frame: np.ndarray, boxes: dict,
        frame_number: int, timestamp: str, clip_id: str
    ) -> list[DetectionRecord]:
        """
        Detects forklifts carrying 3 or more standardized blocks (Section 6.3.2).

        Association: blocks are assigned to the nearest forklift bbox.
        If block count >= FORKLIFT_BLOCK_THRESHOLD → violation.
        """
        violations = []
        rule = self._rule_map[3]

        if not boxes["forklift"]:
            return violations

        for forklift in boxes["forklift"]:
            fx1, fy1, fx2, fy2 = forklift["bbox"]

            # Count blocks whose centroid falls within or near the forklift bbox
            # (expanded by 50px to catch blocks on the extended forks)
            fork_zone_expanded = [fx1 - 50, fy1 - 50, fx2 + 200, fy2 + 50]  # Forks extend forward

            block_count = 0
            for block in boxes["block"]:
                bx1, by1, bx2, by2 = block["bbox"]
                b_cx, b_cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                if (fork_zone_expanded[0] < b_cx < fork_zone_expanded[2] and
                        fork_zone_expanded[1] < b_cy < fork_zone_expanded[3]):
                    block_count += 1

            # When the model does not expose a dedicated block class, estimate
            # the number of block-like objects from the cropped fork region.
            if block_count == 0 or not self._has_custom_compliance_classes:
                heuristic_count = self._estimate_block_count_from_frame(frame, fork_zone_expanded)
                block_count = max(block_count, heuristic_count)

            if block_count >= settings.FORKLIFT_BLOCK_THRESHOLD:
                violations.append(self._make_record(
                    rule=rule,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    clip_id=clip_id,
                    bbox=forklift["bbox"],
                    confidence=forklift["conf"],
                    description=(
                        f"Forklift detected carrying {block_count} standardized blocks. "
                        f"Per Section 6.3.2, the maximum safe load is 2 blocks. "
                        f"Current load exceeds threshold by {block_count - 2} block(s), "
                        f"creating vehicle instability risk. Immediate stop required."
                    )
                ))

        return violations

    def _estimate_block_count_from_frame(self, frame: np.ndarray, zone_bbox: list[int]) -> int:
        """
        Estimate how many block-like objects are present in the forklift fork zone.

        This is a fallback for base COCO models that cannot detect the custom
        `block` class. It looks for compact connected components in the expanded
        fork region and counts rectangular candidates that are plausibly blocks.
        """
        x1, y1, x2, y2 = zone_bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return 0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Evaluate both polarities because blocks may be lighter or darker than
        # the surrounding pallet/floor depending on the video.
        candidates = self._extract_block_candidates_from_binary(binary, crop.shape[:2]) # type: ignore
        inverse_candidates = self._extract_block_candidates_from_binary(cv2.bitwise_not(binary), crop.shape[:2]) # type: ignore

        return max(candidates, inverse_candidates)

    def _extract_block_candidates_from_binary(self, binary: np.ndarray, crop_shape: tuple[int, int]) -> int:
        """Count compact connected components that resemble standardized blocks."""
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
        crop_area = float(crop_shape[0] * crop_shape[1]) or 1.0
        count = 0

        for label_index in range(1, num_labels):
            x = stats[label_index, cv2.CC_STAT_LEFT]
            y = stats[label_index, cv2.CC_STAT_TOP]
            width = stats[label_index, cv2.CC_STAT_WIDTH]
            height = stats[label_index, cv2.CC_STAT_HEIGHT]
            area = stats[label_index, cv2.CC_STAT_AREA]

            if area < 120:
                continue

            area_ratio = area / crop_area
            if not (0.002 <= area_ratio <= 0.18):
                continue

            if width < 10 or height < 10:
                continue

            aspect_ratio = width / float(height)
            if not (0.35 <= aspect_ratio <= 4.0):
                continue

            fill_ratio = area / float(width * height)
            if fill_ratio < 0.35:
                continue

            # Ignore very large components that are likely to be the forklift body.
            if area_ratio > 0.12:
                continue

            # Keep the component roughly in the upper 90% of the fork zone.
            if y > crop_shape[0] * 0.9:
                continue

            count += 1

        return count

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_record(
        self,
        rule: ComplianceRule,
        frame_number: int,
        timestamp: str,
        clip_id: str,
        bbox: list[int],
        confidence: float,
        description: str
    ) -> DetectionRecord:
        """Factory method for creating a DetectionRecord from a rule + detection."""
        return DetectionRecord(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp,
            clip_id=clip_id,
            zone=self._infer_zone(clip_id),
            class_id=rule.class_id,
            behavior_class=rule.unsafe_behavior,
            policy_rule_ref=rule.policy_section_ref,
            event_description=description,
            severity=rule.severity,
            frame_number=frame_number,
            confidence=round(confidence, 3),
            bbox=bbox,
        )

    def _infer_zone(self, clip_id: str) -> str:
        """
        Infer the facility zone from the clip filename.
        Expects filenames like 'zone1_clip03.mp4' or 'cam1_...'.
        Falls back to 'Zone-1' for unknown naming conventions.
        """
        lower = clip_id.lower()
        for i in range(1, 5):
            if f"zone{i}" in lower or f"zone-{i}" in lower or f"cam{i}" in lower:
                return f"Zone-{i}"
        return "Zone-1"

    def _draw_violation(self, frame: np.ndarray, record: DetectionRecord) -> np.ndarray:
        """Draw bounding box and label on the annotated frame."""
        color_map = {
            "HIGH": (0, 165, 255),      # Orange
            "CRITICAL": (0, 0, 255),    # Red
            "MEDIUM": (0, 255, 255),    # Yellow
            "LOW": (0, 255, 0),         # Green
        }
        color = color_map.get(record.severity, (255, 255, 255))
        x1, y1, x2, y2 = record.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{record.behavior_class} [{record.severity}]"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        """Encode a frame as base64 JPEG string for API transfer."""
        import base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
detection_engine = DetectionEngine()
