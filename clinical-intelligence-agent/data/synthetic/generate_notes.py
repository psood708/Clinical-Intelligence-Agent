"""
Generate synthetic clinical notes using local Ollama.
Use this if MIMIC-IV-Note access is pending or you want supplementary data.

Generates realistic but entirely synthetic patient notes covering:
- Diabetes management
- Hypertension / cardiac conditions
- Post-surgical discharge summaries
- Outpatient referral letters

Usage:
    python data/synthetic/generate_notes.py --n 200 --output data/synthetic/notes.jsonl
"""
import json
import argparse
import random
import logging
from pathlib import Path
from datetime import datetime
import ollama
from src.utils.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Templates ──────────────────────────────────────────────────────────────

NOTE_TYPES = [
    "discharge_summary",
    "outpatient_note",
    "referral_letter",
    "emergency_note",
    "follow_up_note",
]

CONDITIONS = [
    "Type 2 Diabetes Mellitus with HbA1c of {hba1c}%",
    "Hypertension, Stage {stage}",
    "Chronic Kidney Disease Stage {stage}",
    "COPD with FEV1 {fev1}%",
    "Heart Failure with EF {ef}%",
    "Atrial Fibrillation",
    "Acute MI, managed with PCI",
    "Sepsis secondary to UTI",
]

GENERATION_PROMPT = """Generate a realistic synthetic clinical {note_type} for a patient with the following profile:
- Age: {age} years old, {sex}
- Primary condition: {condition}
- Comorbidities: {comorbidities}

Write a realistic, detailed clinical note as it would appear in an EHR system.
Include: chief complaint, history of present illness, medications, vitals, assessment, and plan.
Use standard clinical abbreviations (HTN, DM2, SOB, etc.).
Make it realistic but ENTIRELY SYNTHETIC — no real patient data.
Length: 200-400 words."""


def generate_note(client: ollama.Client, note_type: str) -> dict:
    """Generate a single synthetic clinical note."""
    age = random.randint(35, 85)
    sex = random.choice(["male", "female"])
    condition = random.choice(CONDITIONS).format(
        hba1c=round(random.uniform(7.5, 12.0), 1),
        stage=random.randint(1, 4),
        fev1=random.randint(25, 65),
        ef=random.randint(20, 55),
    )
    comorbidities = random.sample(
        ["HTN", "CKD Stage 2", "hypothyroidism", "obesity BMI 32", "depression", "GERD"], k=2
    )

    response = client.chat(
        model=settings.ollama_extractor_model,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                note_type=note_type,
                age=age,
                sex=sex,
                condition=condition,
                comorbidities=", ".join(comorbidities),
            ),
        }],
        options={"temperature": 0.9},  # High temp for variety
    )

    return {
        "doc_id": f"syn_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "note_type": note_type,
        "text": response["message"]["content"],
        "metadata": {
            "source": "synthetic",
            "age": age,
            "sex": sex,
            "primary_condition": condition,
            "comorbidities": comorbidities,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of notes to generate")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/notes.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = ollama.Client(host=settings.ollama_base_url)
    logger.info(f"Generating {args.n} synthetic notes using {settings.ollama_extractor_model}")

    with open(output_path, "w") as f:
        for i in range(args.n):
            note_type = random.choice(NOTE_TYPES)
            try:
                note = generate_note(client, note_type)
                f.write(json.dumps(note) + "\n")
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{args.n} notes")
            except Exception as e:
                logger.error(f"Failed note {i}: {e}")

    logger.info(f"Done — {args.n} notes saved to {output_path}")


if __name__ == "__main__":
    main()
