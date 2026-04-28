"""
BioBERT embeddings for clinical text.
Primary: dmis-lab/biobert-v1.1 (local, no API key needed)
Fallback: nomic-embed-text via Ollama or Cohere Embed 4 (free tier)
"""
import logging
from typing import Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BioBERTEmbedder:
    """
    Clinical-domain embeddings using BioBERT.
    Runs fully locally — zero API cost, zero PHI egress.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.biobert_model
        self.device = torch.device(settings.biobert_device)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            logger.info(f"Loading BioBERT: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            logger.info("BioBERT loaded")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, 768) float32 array."""
        self._load()
        all_embeddings = []

        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i : i + settings.embedding_batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self._model(**encoded)
                # CLS token embedding — standard for BioBERT sentence representation
                embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
