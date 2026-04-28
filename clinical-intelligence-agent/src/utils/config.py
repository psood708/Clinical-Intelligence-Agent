"""
Centralised configuration management.
All settings loaded from environment variables via .env file.
"""
from functools import lru_cache
from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class RoutingStrategy(str, Enum):
    PRIVACY_FIRST = "privacy_first"   # Ollama only — no egress
    SPEED_FIRST = "speed_first"       # Prefer Groq
    HYBRID = "hybrid"                 # Smart routing (default)


class Settings(BaseSettings):
    # ── Ollama ────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_extractor_model: str = "qwen2.5:14b"
    ollama_verifier_model: str = "llama3.3:70b"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── Free hosted APIs ──────────────────────
    groq_api_key: str = ""
    groq_synthesizer_model: str = "llama-3.3-70b-versatile"

    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"
    cohere_embed_model: str = "embed-english-v3.0"

    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    openrouter_api_key: str = ""

    # ── Retrieval ─────────────────────────────
    hnsw_index_path: str = "data/indexes/clinical_hnsw.bin"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # ── BioBERT ───────────────────────────────
    biobert_model: str = "dmis-lab/biobert-v1.1"
    biobert_device: str = "mps"
    embedding_batch_size: int = 32
    embedding_dimension: int = 768

    # ── Observability ─────────────────────────
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # ── Evaluation ────────────────────────────
    eval_judge_model: str = "ollama/llama3.3:70b"
    eval_dataset_path: str = "data/eval/gold_standard.json"
    eval_batch_size: int = 10

    # ── Application ───────────────────────────
    log_level: str = "INFO"
    environment: str = "development"
    max_document_size_mb: int = 10
    supported_formats: str = "pdf,txt,docx"
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
