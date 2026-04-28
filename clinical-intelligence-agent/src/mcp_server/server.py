"""
MCP Server — exposes the Clinical Intelligence Agent as a set of tools
that any MCP-compatible client can use (Claude Desktop, Cursor, custom apps).

Tools exposed:
  - extract_clinical_entities  → structured extraction from text
  - search_similar_cases       → HNSW + reranking over clinical corpus
  - summarize_document         → full pipeline (extract + verify + synthesize)
  - verify_extraction          → challenge an extraction with evidence

To test with Claude Desktop, add to your MCP config:
  {
    "clinical-agent": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/clinical-intelligence-agent"
    }
  }
"""
import logging
from pathlib import Path
from fastmcp import FastMCP
from src.agents.extractor import ExtractorAgent, ClinicalExtraction
from src.agents.orchestrator import ClinicalPipeline
from src.retrieval.hnsw_index import ClinicalHNSWIndex
from src.retrieval.embeddings import BioBERTEmbedder
from src.retrieval.reranker import ClinicalReranker
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

mcp = FastMCP("clinical-intelligence-agent")

# Lazily initialised singletons — don't pay startup cost until first call
_pipeline: ClinicalPipeline | None = None
_extractor: ExtractorAgent | None = None


def _get_pipeline() -> ClinicalPipeline:
    global _pipeline
    if _pipeline is None:
        index_path = Path(settings.hnsw_index_path)
        if index_path.exists():
            index = ClinicalHNSWIndex.load(index_path)
            logger.info(f"Loaded HNSW index from {index_path}")
        else:
            logger.warning("HNSW index not found — retrieval will return empty results")
            index = ClinicalHNSWIndex()
        _pipeline = ClinicalPipeline(
            hnsw_index=index,
            embedder=BioBERTEmbedder(),
            reranker=ClinicalReranker(),
        )
    return _pipeline


def _get_extractor() -> ExtractorAgent:
    global _extractor
    if _extractor is None:
        _extractor = ExtractorAgent()
    return _extractor


# ─── Tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def extract_clinical_entities(clinical_text: str) -> dict:
    """
    Extract structured clinical entities from unstructured clinical text.

    Args:
        clinical_text: Raw clinical note, discharge summary, or referral text.

    Returns:
        Structured JSON with medications, conditions, procedures, lab values,
        risk flags, and confidence score.
    """
    extraction = _get_extractor().extract(clinical_text)
    return extraction.model_dump()


@mcp.tool()
def search_similar_cases(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the clinical corpus for cases similar to the query.
    Uses HNSW vector search with BioBERT embeddings + Cohere reranking.

    Args:
        query: Clinical query text (chief complaint, condition name, or note excerpt).
        top_k: Number of similar cases to return (max 20).

    Returns:
        List of similar cases with text, metadata, and similarity scores.
    """
    top_k = min(top_k, 20)
    pipeline = _get_pipeline()
    query_emb = pipeline.embedder.embed_single(query)
    candidates = pipeline.index.search(query_emb, k=max(top_k * 2, 20))
    reranked = pipeline.reranker.rerank(query, candidates, top_k=top_k)

    return [
        {
            "rank": doc.rank,
            "similarity_score": doc.similarity_score,
            "text": doc.text[:800],
            "metadata": doc.metadata,
        }
        for doc in reranked
    ]


@mcp.tool()
def summarize_document(clinical_text: str, document_id: str = "doc") -> dict:
    """
    Run the full clinical intelligence pipeline on a document.
    Extracts → retrieves similar cases → verifies → synthesizes.

    Args:
        clinical_text: Complete clinical document text.
        document_id: Optional identifier for the document.

    Returns:
        Comprehensive clinical summary with confidence score and warnings.
    """
    summary = _get_pipeline().run(clinical_text, document_id)
    return summary.model_dump()


@mcp.tool()
def get_system_status() -> dict:
    """
    Check the system status: index size, model availability, routing strategy.

    Returns:
        Status dict showing which components are ready.
    """
    pipeline = _get_pipeline()
    return {
        "index_size": len(pipeline.index.nodes),
        "index_path": settings.hnsw_index_path,
        "routing_strategy": settings.routing_strategy.value,
        "groq_available": bool(settings.groq_api_key),
        "cohere_available": bool(settings.cohere_api_key),
        "ollama_models": {
            "extractor": settings.ollama_extractor_model,
            "verifier": settings.ollama_verifier_model,
        },
    }


if __name__ == "__main__":
    mcp.run()
