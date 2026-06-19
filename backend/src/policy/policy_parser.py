"""
policy_parser.py — Module 1 Support: Extract structured compliance rules from the
Occupational Health & Safety Compliance Policy Manual using Groq LLM.

Design decisions:
- We embed the known policy text as a fallback constant (POLICY_TEXT) so the system
  works even if the PDF is not present at runtime. The PDF path is also read if available.
- Severity is derived from the policy's own alert callout language:
    "WARNING"                → HIGH
    "CRITICAL SAFETY NOTICE" → CRITICAL
- Output is a Python dataclass (ComplianceRule) consumed by the Detection Engine.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from groq import Groq

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model for a single extracted compliance rule
# ---------------------------------------------------------------------------

@dataclass
class ComplianceRule:
    class_id: int                   # 0-3, matching Section 8 Class IDs in the policy
    behavior_domain: str            # e.g. "Pedestrian Movement"
    unsafe_behavior: str            # e.g. "Safe Walkway Violation"
    safe_behavior: str              # e.g. "Safe Walkway"
    observable_indicator: str       # Key visual cue for the detector
    policy_section_ref: str         # e.g. "Section 3.3.2"
    hazard_context: str             # Summary of why this is dangerous
    severity: str                   # LOW / MEDIUM / HIGH / CRITICAL
    alert_callout_type: str         # "WARNING" or "CRITICAL SAFETY NOTICE"
    escalation_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedded policy text (condensed key sections) — used for LLM context
# This avoids needing PDF parsing libs and ensures faithful grounding.
# ---------------------------------------------------------------------------

POLICY_SUMMARY_FOR_LLM = """
KAFAOGLU METAL PLASTIK — OHS COMPLIANCE POLICY MANUAL (KMP-OHS-POL-001)

SECTION 8 QUICK REFERENCE — BEHAVIOR CLASSES:
Class 0: Pedestrian Movement | Unsafe: Safe Walkway Violation | Safe: Safe Walkway
  Observable indicator: Person outside green floor markings
  Policy section for unsafe: 3.3.2

Class 1: Equipment Interaction | Unsafe: Unauthorized Intervention | Safe: Authorized Intervention
  Observable indicator: Person interacting with equipment without green vest or required safety equipment
  Policy section for unsafe: 4.3.2

Class 2: Electrical Safety | Unsafe: Opened Panel Cover | Safe: Closed Panel Cover
  Observable indicator: Panel cover in open position during production operations
  Policy section for unsafe: 5.2.2

Class 3: Forklift Load | Unsafe: Carrying Overload with Forklift | Safe: Safe Carrying
  Observable indicator: 3 or more standardized blocks on forklift forks
  Policy section for unsafe: 6.3.2

ALERT CALLOUT LEVELS IN THE POLICY DOCUMENT:
- Section 3.3 contains a "WARNING" callout: "The Safe Walkway Violation is the highest-frequency 
  unsafe behavior recorded at this facility."
- Section 4.3 contains a "CRITICAL SAFETY NOTICE": "Any person seen interacting with equipment 
  who is not wearing the green vest must be assumed to be performing an Unauthorized Intervention."
- Section 5.2 contains a "WARNING" callout: "Leaving a panel cover open — even when doing so 
  feels like a minor or temporary oversight — is classified as an unsafe behavior."
- Section 6.3 contains a "CRITICAL SAFETY NOTICE": "The block count threshold is unambiguous: 
  two blocks or fewer is safe; three blocks or more is an overload."

VEST SYSTEM:
- Green safety vest → Authorized personnel (safe for equipment interaction)
- Red-black safety vest → General production floor personnel (NOT authorized for equipment intervention)

FORKLIFT THRESHOLDS:
- 2 blocks or fewer = COMPLIANT (Safe Carrying)
- 3 blocks or more = NON-COMPLIANT (Carrying Overload with Forklift)
"""

# ---------------------------------------------------------------------------
# Severity derivation (hardcoded from policy — not LLM-generated)
# LLM is used only for hazard context and descriptions, NOT for safety-critical
# severity mappings. Severity comes from the policy callout language directly.
# ---------------------------------------------------------------------------

SEVERITY_MAP = {
    0: {  # Safe Walkway Violation
        "severity": "HIGH",
        "alert_callout_type": "WARNING",
        "reason": "Highest-frequency unsafe behavior; places person near forklift/machinery hazards (Section 3.3 WARNING callout)"
    },
    1: {  # Unauthorized Intervention
        "severity": "CRITICAL",
        "alert_callout_type": "CRITICAL SAFETY NOTICE",
        "reason": "Direct injury risk from unauthorized equipment contact; green vest is the unambiguous authorization indicator (Section 4.3 CRITICAL SAFETY NOTICE)"
    },
    2: {  # Opened Panel Cover
        "severity": "HIGH",
        "alert_callout_type": "WARNING",
        "reason": "Exposes live electrical components; classified as unsafe regardless of duration or proximity (Section 5.2 WARNING callout)"
    },
    3: {  # Carrying Overload with Forklift
        "severity": "CRITICAL",
        "alert_callout_type": "CRITICAL SAFETY NOTICE",
        "reason": "Vehicle instability risk with unambiguous threshold; 3+ blocks triggers immediate response (Section 6.3 CRITICAL SAFETY NOTICE)"
    }
}

# ---------------------------------------------------------------------------
# Static rule definitions (grounded in the policy, verified manually)
# The LLM enriches these with hazard context; it does NOT define the rules.
# ---------------------------------------------------------------------------

STATIC_RULES = [
    ComplianceRule(
        class_id=0,
        behavior_domain="Pedestrian Movement",
        unsafe_behavior="Safe Walkway Violation",
        safe_behavior="Safe Walkway",
        observable_indicator="Person's position outside green-painted floor marking boundaries",
        policy_section_ref="Section 3.3.2",
        hazard_context="",  # Filled by LLM
        severity=SEVERITY_MAP[0]["severity"],
        alert_callout_type=SEVERITY_MAP[0]["alert_callout_type"],
        escalation_keywords=["walkway", "green marking", "pedestrian zone", "outside boundary"]
    ),
    ComplianceRule(
        class_id=1,
        behavior_domain="Equipment Interaction",
        unsafe_behavior="Unauthorized Intervention",
        safe_behavior="Authorized Intervention",
        observable_indicator="Person interacting with equipment wearing red-black vest or no green vest",
        policy_section_ref="Section 4.3.2",
        hazard_context="",  # Filled by LLM
        severity=SEVERITY_MAP[1]["severity"],
        alert_callout_type=SEVERITY_MAP[1]["alert_callout_type"],
        escalation_keywords=["green vest", "red vest", "equipment", "unauthorized", "intervention"]
    ),
    ComplianceRule(
        class_id=2,
        behavior_domain="Electrical Safety",
        unsafe_behavior="Opened Panel Cover",
        safe_behavior="Closed Panel Cover",
        observable_indicator="Electrical panel cover in open position during production operations",
        policy_section_ref="Section 5.2.2",
        hazard_context="",  # Filled by LLM
        severity=SEVERITY_MAP[2]["severity"],
        alert_callout_type=SEVERITY_MAP[2]["alert_callout_type"],
        escalation_keywords=["panel", "electrical", "open cover", "protective barrier"]
    ),
    ComplianceRule(
        class_id=3,
        behavior_domain="Forklift Load Management",
        unsafe_behavior="Carrying Overload with Forklift",
        safe_behavior="Safe Carrying",
        observable_indicator="3 or more standardized blocks on forklift forks during any phase of operation",
        policy_section_ref="Section 6.3.2",
        hazard_context="",  # Filled by LLM
        severity=SEVERITY_MAP[3]["severity"],
        alert_callout_type=SEVERITY_MAP[3]["alert_callout_type"],
        escalation_keywords=["forklift", "blocks", "overload", "three blocks", "load capacity"]
    ),
]


# ---------------------------------------------------------------------------
# PolicyParser — uses Groq LLM to enrich hazard context descriptions
# ---------------------------------------------------------------------------

class PolicyParser:
    """
    Parses the OHS Compliance Policy Manual and returns structured ComplianceRule objects.

    Strategy:
    1. Use STATIC_RULES as the authoritative source for safety-critical fields
       (class_id, severity, policy_section_ref, observable_indicator).
    2. Use Groq LLM ONLY to generate human-readable hazard_context descriptions
       that will appear in violation reports.
    3. This prevents hallucination from corrupting safety logic.
    """

    def __init__(self):
        # If no API key is configured, avoid creating the Groq client
        # and skip LLM enrichment to prevent confusing connection errors.
        if not settings.GROQ_API_KEY:
            self.client = None
            logger.info("GROQ_API_KEY not set; skipping LLM enrichment for policy rules.")
        else:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        self._rules: Optional[list[ComplianceRule]] = None

    def get_rules(self) -> list[ComplianceRule]:
        """Returns parsed rules, loading from LLM on first call."""
        if self._rules is None:
            self._rules = self._load_rules()
        return self._rules

    def _load_rules(self) -> list[ComplianceRule]:
        """Enriches static rules with LLM-generated hazard context."""
        logger.info("Enriching compliance rules with Groq LLM hazard context...")
        rules = [ComplianceRule(**asdict(r)) for r in STATIC_RULES]  # Deep copy

        # If client wasn't initialized (no API key), skip enrichment.
        if self.client is None:
            logger.info("Skipping LLM enrichment: no Groq client available. Using static rules.")
            return rules

        try:
            enriched = self._enrich_with_llm(rules)
            logger.info("Policy rules successfully enriched by LLM.")
            return enriched
        except Exception as e:
            logger.warning(f"LLM enrichment failed ({e}). Using static rules without hazard context.")
            return rules

    def _enrich_with_llm(self, rules: list[ComplianceRule]) -> list[ComplianceRule]:
        """
        Calls Groq to generate concise hazard context for each rule.
        Instructs the model strictly — it must return JSON only.
        """
        rule_list_for_prompt = [
            {
                "class_id": r.class_id,
                "unsafe_behavior": r.unsafe_behavior,
                "behavior_domain": r.behavior_domain,
                "policy_section_ref": r.policy_section_ref,
            }
            for r in rules
        ]

        system_prompt = """You are a compliance document analyst for an industrial safety system.
You will be given a summary of an Occupational Health & Safety policy and a list of 4 violation classes.
Your task: for each class, write a concise 1-2 sentence hazard_context explaining WHY this behavior is 
dangerous in a factory setting.

Return ONLY a valid JSON array with this exact structure:
[
  {"class_id": 0, "hazard_context": "..."},
  {"class_id": 1, "hazard_context": "..."},
  {"class_id": 2, "hazard_context": "..."},
  {"class_id": 3, "hazard_context": "..."}
]
No preamble, no markdown, no explanation. ONLY the JSON array."""

        user_prompt = f"""POLICY SUMMARY:
{POLICY_SUMMARY_FOR_LLM}

VIOLATION CLASSES TO ENRICH:
{json.dumps(rule_list_for_prompt, indent=2)}

Return the JSON array with hazard_context for each class_id."""

        response = self.client.chat.completions.create( # type: ignore
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800,
        )

        raw = response.choices[0].message.content.strip() # type: ignore

        # Strip markdown fences if the model wrapped it anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        enriched_data = json.loads(raw)

        # Validate and apply — only update hazard_context, never severity/refs
        context_map = {item["class_id"]: item["hazard_context"] for item in enriched_data}
        for rule in rules:
            if rule.class_id in context_map:
                rule.hazard_context = context_map[rule.class_id]

        return rules

    def get_rule_by_class_id(self, class_id: int) -> Optional[ComplianceRule]:
        """Look up a rule by its policy class ID (0-3)."""
        for rule in self.get_rules():
            if rule.class_id == class_id:
                return rule
        return None

    def get_rules_as_dict(self) -> list[dict]:
        """Return all rules serialized to dicts (for API responses)."""
        return [asdict(r) for r in self.get_rules()]

    def get_severity_for_class(self, class_id: int) -> str:
        """Direct severity lookup — uses static map, not LLM."""
        return SEVERITY_MAP.get(class_id, {}).get("severity", "MEDIUM")


# ---------------------------------------------------------------------------
# Singleton instance used across the application
# ---------------------------------------------------------------------------
policy_parser = PolicyParser()
