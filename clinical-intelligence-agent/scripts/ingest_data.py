"""
Ingest clinical documents and build the HNSW index.
Run this once before starting the MCP server.

Usage:
    # From synthetic data
    python scripts/ingest_data.py --source data/synthetic/notes.jsonl

    # From MIMIC-IV-Note (after download)
    python scripts/ingest_data.py --source data/mimic/discharge.csv --format mimic
"""
import json
import csv
import argparse
import logging
from pathlib import Path
import numpy as np
from rich.progress import track
from src.retrieval.hnsw_index import ClinicalHNSWIndex
from src.retrieval.embeddings import BioBERTEmbedder
from src.utils.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def load_mimic(path: Path) -> list[dict]:
    """Load MIMIC-IV-Note discharge summaries."""
    docs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            docs.append({
                "doc_id": f"mimic_{row.get('note_id', i)}",
                "text": row.get("text", ""),
                "metadata": {
                    "source": "mimic-iv",
                    "subject_id": row.get("subject_id", ""),
                    "note_type": row.get("note_type", "discharge"),
                },
            })
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to source data file")
    parser.add_argument(
        "--format", default="jsonl", choices=["jsonl", "mimic"], help="Source format"
    )
    parser.add_argument(
        "--output", default=settings.hnsw_index_path, help="Output index path"
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    # Load documents
    logger.info(f"Loading from {source_path}")
    if args.format == "jsonl":
        documents = load_jsonl(source_path)
    else:
        documents = load_mimic(source_path)

    if args.max_docs:
        documents = documents[: args.max_docs]

    # Filter empties
    documents = [d for d in documents if len(d.get("text", "")) > 50]
    logger.info(f"Loaded {len(documents)} documents")

    # Embed
    embedder = BioBERTEmbedder()
    texts = [d["text"] for d in documents]
    logger.info(f"Embedding {len(texts)} texts with BioBERT (device={settings.biobert_device})")

    all_embeddings = []
    batch_size = settings.embedding_batch_size
    for i in track(range(0, len(texts), batch_size), description="Embedding"):
        batch = texts[i : i + batch_size]
        embs = embedder.embed(batch)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build index
    index = ClinicalHNSWIndex()
    index.build(documents, embeddings)
    index.save(Path(args.output))
    logger.info(f"Index saved → {args.output} ({len(documents)} vectors)")


if __name__ == "__main__":
    main()
