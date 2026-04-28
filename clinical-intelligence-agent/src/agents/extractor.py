"""
Extractor Agent — structured clinical entity extraction.
Model: qwen3.5:9b via Ollama (local, free, fast)
Input:  Raw clinical text (notes, discharge summaries, referrals)
Output: Validated ClinicalExtraction Pydantic model

Design decision: Use qwen3.5:9b for extraction because:
1. Extraction is a narrow, well-defined structured output task — 9B is enough
2. Qwen3 series is specifically strong at structured JSON output
3. At 6.6 GB it runs fast on M-series Mac, zero PHI egress
"""
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
import ollama
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Output Schema ──────────────────────────────────────────────────────────

class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None

class Condition(BaseModel):
    name: str
    status: str = Field(description="active | resolved | chronic | suspected")
    icd_hint: Optional[str] = None

class FollowUp(BaseModel):
    action: str
    timeframe: Optional[str] = None
    responsible_party: Optional[str] = None

class RiskFlag(BaseModel):
    flag: str
    severity: str = Field(description="low | medium | high | critical")
    rationale: str

class ClinicalExtraction(BaseModel):
    """Structured output of the Extractor agent."""
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    chief_complaint: Optional[str] = None
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    lab_values: dict[str, str] = Field(default_factory=dict)
    allergies: list[str] = Field(default_factory=list)
    follow_ups: list[FollowUp] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    raw_text_length: int = 0


# ─── System Prompt ──────────────────────────────────────────────────────────

EXTRACTOR_SYSTEM = """You are a precise clinical information extractor.
Extract ALL clinical entities from the provided text and return them as valid JSON.
Be thorough — missing a medication or risk flag is worse than including an uncertain one.
Never hallucinate information not present in the text.
Return ONLY the JSON object, no preamble, no markdown."""

EXTRACTOR_TEMPLATE = """Extract all clinical information from this text.

TEXT:
{text}

Return a JSON object with this exact structure:
{{
  "patient_age": "string or null",
  "patient_sex": "string or null",
  "chief_complaint": "string or null",
  "medications": [{{"name": "...", "dose": "...", "frequency": "...", "route": "..."}}],
  "conditions": [{{"name": "...", "status": "active|resolved|chronic|suspected", "icd_hint": "..."}}],
  "procedures": ["..."],
  "lab_values": {{"test_name": "value unit"}},
  "allergies": ["..."],
  "follow_ups": [{{"action": "...", "timeframe": "...", "responsible_party": "..."}}],
  "risk_flags": [{{"flag": "...", "severity": "low|medium|high|critical", "rationale": "..."}}],
  "extraction_confidence": 0.0
}}"""


# ─── Agent ──────────────────────────────────────────────────────────────────

class ExtractorAgent:
    """
    Extracts structured clinical entities from unstructured text.
    Runs entirely on local Ollama — no API key, no PHI egress.
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.ollama_extractor_model
        self._client = ollama.Client(host=settings.ollama_base_url)

    def extract(self, text: str) -> ClinicalExtraction:
        """
        Extract clinical entities from text.
        Returns a validated ClinicalExtraction or raises on parse failure.
        """
        prompt = EXTRACTOR_TEMPLATE.format(text=text[:4000])  # 4K char limit per note

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1},  # Low temp for deterministic extraction
            )

            raw_json = response["message"]["content"].strip()
            # Strip markdown code blocks if model wraps them
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]

            data = json.loads(raw_json)
            extraction = ClinicalExtraction(**data, raw_text_length=len(text))
            logger.debug(
                f"Extraction: {len(extraction.conditions)} conditions, "
                f"{len(extraction.medications)} meds, "
                f"{len(extraction.risk_flags)} risk flags"
            )
            return extraction

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return ClinicalExtraction(
                extraction_confidence=0.0,
                raw_text_length=len(text),
                risk_flags=[RiskFlag(
                    flag="extraction_failed",
                    severity="medium",
                    rationale=f"JSON parse error: {e}"
                )],
            )
