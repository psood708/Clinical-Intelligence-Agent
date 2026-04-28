"""
Unit tests for the Extractor agent.
Uses mock Ollama responses to test parsing logic without requiring a running model.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.agents.extractor import ExtractorAgent, ClinicalExtraction


SAMPLE_NOTE = """
Patient: 67-year-old male presenting with increasing shortness of breath and bilateral
lower extremity edema for 3 days. PMH: HTN, DM2, CKD Stage 3.

Medications: Metformin 1000mg BID, Lisinopril 10mg daily, Furosemide 40mg daily.
Allergies: Penicillin (rash).

Vitals: BP 156/92, HR 88, SpO2 94% on RA.
Labs: Creatinine 2.1 (baseline 1.8), BNP 850.

Assessment: Acute decompensated heart failure. HbA1c 8.4%, poorly controlled DM2.

Plan:
1. IV Lasix 80mg bolus, monitor I&O
2. Restrict fluids to 1.5L/day
3. Echo in AM
4. Nephrology consult for worsening CKD
5. Follow up cardiology in 2 weeks
"""

MOCK_EXTRACTION_JSON = """{
  "patient_age": "67",
  "patient_sex": "male",
  "chief_complaint": "shortness of breath and bilateral lower extremity edema",
  "medications": [
    {"name": "Metformin", "dose": "1000mg", "frequency": "BID", "route": "oral"},
    {"name": "Lisinopril", "dose": "10mg", "frequency": "daily", "route": "oral"},
    {"name": "Furosemide", "dose": "40mg", "frequency": "daily", "route": "oral"}
  ],
  "conditions": [
    {"name": "Acute decompensated heart failure", "status": "active", "icd_hint": "I50.9"},
    {"name": "Hypertension", "status": "chronic", "icd_hint": "I10"},
    {"name": "Type 2 Diabetes Mellitus", "status": "active", "icd_hint": "E11"},
    {"name": "CKD Stage 3", "status": "chronic", "icd_hint": "N18.3"}
  ],
  "procedures": ["IV Lasix bolus", "Echo", "Nephrology consult"],
  "lab_values": {"Creatinine": "2.1", "BNP": "850", "HbA1c": "8.4%"},
  "allergies": ["Penicillin"],
  "follow_ups": [
    {"action": "Cardiology follow-up", "timeframe": "2 weeks", "responsible_party": "Cardiology"},
    {"action": "Nephrology consult", "timeframe": "inpatient", "responsible_party": "Nephrology"}
  ],
  "risk_flags": [
    {"flag": "Worsening renal function", "severity": "high", "rationale": "Creatinine 2.1 above baseline 1.8"},
    {"flag": "Poorly controlled diabetes", "severity": "medium", "rationale": "HbA1c 8.4%"}
  ],
  "extraction_confidence": 0.88
}"""


class TestExtractorAgent:

    @patch("ollama.Client")
    def test_basic_extraction(self, mock_client_class):
        """Extracted fields match the clinical note content."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": MOCK_EXTRACTION_JSON}
        }
        mock_client_class.return_value = mock_client

        agent = ExtractorAgent()
        result = agent.extract(SAMPLE_NOTE)

        assert isinstance(result, ClinicalExtraction)
        assert result.patient_age == "67"
        assert result.patient_sex == "male"
        assert len(result.medications) == 3
        assert len(result.conditions) == 4
        assert "Penicillin" in result.allergies

    @patch("ollama.Client")
    def test_medication_names(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": MOCK_EXTRACTION_JSON}}
        mock_client_class.return_value = mock_client

        agent = ExtractorAgent()
        result = agent.extract(SAMPLE_NOTE)
        med_names = [m.name for m in result.medications]

        assert "Metformin" in med_names
        assert "Lisinopril" in med_names
        assert "Furosemide" in med_names

    @patch("ollama.Client")
    def test_risk_flags_present(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": MOCK_EXTRACTION_JSON}}
        mock_client_class.return_value = mock_client

        agent = ExtractorAgent()
        result = agent.extract(SAMPLE_NOTE)

        assert len(result.risk_flags) >= 1
        severities = [f.severity for f in result.risk_flags]
        assert any(s in ("high", "critical") for s in severities)

    @patch("ollama.Client")
    def test_handles_malformed_json(self, mock_client_class):
        """Malformed JSON returns graceful error extraction, not an exception."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "This is not JSON at all."}
        }
        mock_client_class.return_value = mock_client

        agent = ExtractorAgent()
        result = agent.extract("some text")

        assert isinstance(result, ClinicalExtraction)
        assert result.extraction_confidence == 0.0
        assert any("extraction_failed" in f.flag for f in result.risk_flags)

    @patch("ollama.Client")
    def test_markdown_stripped(self, mock_client_class):
        """JSON wrapped in markdown code blocks is parsed correctly."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "```json\n" + MOCK_EXTRACTION_JSON + "\n```"}
        }
        mock_client_class.return_value = mock_client

        agent = ExtractorAgent()
        result = agent.extract(SAMPLE_NOTE)

        assert result.patient_age == "67"
