"""
Synthesizer Agent — final clinical summary with source citations.
Model: Llama 3.3 70B via Groq (free: ~500K tokens/day, 315 tok/s)
Fallback: Llama 3.3 70B via Ollama (local, unlimited, slower)

Why Groq here specifically:
- Groq runs at 315 tokens/second — synthesis step goes from 8s → <1s
- This is the user-facing output so quality + speed both matter
- Groq's free tier is more than enough for portfolio-scale usage
"""
import logging
from pydantic import BaseModel
from groq import Groq
import ollama
from src.agents.extractor import ClinicalExtraction
from src.agents.verifier import VerificationResult
from src.retrieval.hnsw_index import RetrievedDocument
from src.utils.config import get_settings, RoutingStrategy

logger = logging.getLogger(__name__)
settings = get_settings()


class ClinicalSummary(BaseModel):
    """Final structured clinical summary."""
    headline: str
    clinical_summary: str
    key_findings: list[str]
    active_problems: list[str]
    medication_review: str
    risk_assessment: str
    recommended_actions: list[str]
    confidence_score: float
    data_sources_used: list[str]
    warnings: list[str]


SYNTHESIZER_SYSTEM = """You are a senior clinical documentation specialist.
Produce clear, accurate, citation-backed clinical summaries.
Every claim must be traceable to the source text or retrieved evidence.
Flag any uncertainty explicitly."""

SYNTHESIZER_TEMPLATE = """Synthesize a comprehensive clinical summary.

VERIFIED EXTRACTION:
{extraction}

VERIFICATION NOTES:
{verification_notes}

SUPPORTING EVIDENCE FROM SIMILAR CASES:
{evidence}

Produce a clinical summary with:
- Headline (one sentence)
- Clinical narrative (2-3 paragraphs)
- Bulleted key findings
- Active problem list
- Medication review
- Risk assessment
- Recommended actions
- Confidence score (0-1)
- Any warnings or uncertainties

Be concise. Cite sources (e.g., "per retrieved Case #2"). Flag uncertainty."""


class SynthesizerAgent:
    """
    Synthesizes verified extractions into a final clinical summary.
    Prefers Groq (fast, free) — falls back to local Ollama.
    """

    def __init__(self):
        self._groq: Groq | None = None
        self._ollama = ollama.Client(host=settings.ollama_base_url)

    def _get_groq(self) -> Groq:
        if self._groq is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._groq = Groq(api_key=settings.groq_api_key)
        return self._groq

    def synthesize(
        self,
        extraction: ClinicalExtraction,
        verification: VerificationResult,
        evidence: list[RetrievedDocument],
    ) -> ClinicalSummary:
        """
        Produce final clinical summary from verified extraction + evidence.
        """
        prompt = self._build_prompt(extraction, verification, evidence)

        # Route based on strategy and API key availability
        use_groq = (
            settings.routing_strategy != RoutingStrategy.PRIVACY_FIRST
            and bool(settings.groq_api_key)
        )

        if use_groq:
            return self._synthesize_groq(prompt, extraction, verification)
        else:
            return self._synthesize_local(prompt, extraction, verification)

    def _synthesize_groq(
        self,
        prompt: str,
        extraction: ClinicalExtraction,
        verification: VerificationResult,
    ) -> ClinicalSummary:
        try:
            client = self._get_groq()
            response = client.chat.completions.create(
                model=settings.groq_synthesizer_model,
                messages=[
                    {"role": "system", "content": SYNTHESIZER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            logger.info("Synthesized via Groq")
            return self._parse_synthesis(content, extraction, verification)

        except Exception as e:
            logger.warning(f"Groq synthesis failed ({e}), falling back to Ollama")
            return self._synthesize_local(prompt, extraction, verification)

    def _synthesize_local(
        self,
        prompt: str,
        extraction: ClinicalExtraction,
        verification: VerificationResult,
    ) -> ClinicalSummary:
        response = self._ollama.chat(
            model=settings.ollama_verifier_model,  # Reuse 70B for synthesis
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        content = response["message"]["content"]
        logger.info("Synthesized via local Ollama")
        return self._parse_synthesis(content, extraction, verification)

    def _parse_synthesis(
        self,
        content: str,
        extraction: ClinicalExtraction,
        verification: VerificationResult,
    ) -> ClinicalSummary:
        """Parse free-text synthesis into ClinicalSummary."""
        # Compute final confidence
        final_confidence = max(
            0.0,
            min(1.0, extraction.extraction_confidence + verification.confidence_adjustment),
        )
        warnings = [f.issue for f in verification.flags if f.severity in ("major", "critical")]
        return ClinicalSummary(
            headline=self._extract_section(content, "headline", "Clinical Summary"),
            clinical_summary=content[:2000],
            key_findings=self._extract_bullets(content, "key findings"),
            active_problems=[c.name for c in extraction.conditions if c.status == "active"],
            medication_review=self._extract_section(content, "medication", "See extraction"),
            risk_assessment=self._extract_section(content, "risk", "See risk flags"),
            recommended_actions=self._extract_bullets(content, "actions"),
            confidence_score=final_confidence,
            data_sources_used=["original_document", "hnsw_retrieval", "cohere_rerank"],
            warnings=warnings,
        )

    def _build_prompt(
        self,
        extraction: ClinicalExtraction,
        verification: VerificationResult,
        evidence: list[RetrievedDocument],
    ) -> str:
        evidence_text = "\n\n".join(
            f"Case #{i + 1} (score={doc.similarity_score:.2f}): {doc.text[:400]}"
            for i, doc in enumerate(evidence[:3])
        )
        ver_notes = "\n".join(
            f"- [{f.severity.upper()}] {f.field}: {f.issue}"
            for f in verification.flags
        ) or "No issues found."

        return SYNTHESIZER_TEMPLATE.format(
            extraction=extraction.model_dump_json(indent=2),
            verification_notes=ver_notes,
            evidence=evidence_text or "No similar cases retrieved.",
        )

    def _extract_section(self, text: str, keyword: str, default: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                return " ".join(lines[i : i + 3]).strip() or default
        return default

    def _extract_bullets(self, text: str, section: str) -> list[str]:
        lines = text.split("\n")
        in_section = False
        bullets = []
        for line in lines:
            if section.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if stripped.startswith(("-", "•", "*", "·")):
                    bullets.append(stripped.lstrip("-•*· ").strip())
                elif stripped and not stripped[0].isdigit() and len(bullets) > 0:
                    break
        return bullets or []
