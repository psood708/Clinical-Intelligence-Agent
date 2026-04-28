"""
BioBERT embeddings for clinical text.
Primary: dmis-lab/biobert-v1.1 (local via HuggingFace, no API key)
Local fallback: nomic-embed-text:v1.5 via Ollama (already on your machine)

BioBERT is preferred for clinical text — pretrained on PubMed and clinical
notes, it understands clinical synonyms that generic embeddings miss.
nomic-embed-text:v1.5 is the zero-setup fallback if BioBERT load fails.
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


class NomicEmbedder:
    """
    Local embeddings using nomic-embed-text:v1.5 via Ollama.
    Zero-setup fallback — already on your machine.
    Produces 768-dim vectors, same dimension as BioBERT.

    Use this when:
    - BioBERT is slow to load and you want faster iteration
    - You're on a machine without enough RAM for BioBERT + agent models
    - Testing the pipeline end-to-end before committing to BioBERT
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.ollama_embedding_model  # nomic-embed-text:v1.5
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=settings.ollama_base_url)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, 768) float32 array."""
        client = self._get_client()
        embeddings = []
        for text in texts:
            response = client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(prefer_local: bool = False):
    """
    Factory — returns the best available embedder.
    prefer_local=True skips BioBERT download and uses Ollama nomic directly.
    """
    if prefer_local:
        logger.info("Using NomicEmbedder (nomic-embed-text:v1.5 via Ollama)")
        return NomicEmbedder()
    try:
        embedder = BioBERTEmbedder()
        logger.info("Using BioBERTEmbedder (clinical domain embeddings)")
        return embedder
    except Exception as e:
        logger.warning(f"BioBERT unavailable ({e}), falling back to NomicEmbedder")
        return NomicEmbedder()
