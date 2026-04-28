"""
Reranking layer — applied after HNSW retrieval to improve precision.
Primary: Cohere Rerank 3.5 (free: 1,000 req/month — more than enough for dev)
Fallback: Cross-encoder via sentence-transformers (local, free)

Why reranking matters:
HNSW retrieval is fast but ANN (approximate) — reranking the top-20
candidates with a cross-encoder model consistently adds 8-15% precision
over embedding similarity alone. For clinical extraction, every point matters.
"""
import logging
from typing import Optional
import cohere
from src.retrieval.hnsw_index import RetrievedDocument
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ClinicalReranker:

    def __init__(self):
        self._cohere: Optional[cohere.Client] = None
        self._local_model = None

    def _get_cohere(self) -> cohere.Client:
        if self._cohere is None:
            if not settings.cohere_api_key:
                raise ValueError("COHERE_API_KEY not set — use local fallback")
            self._cohere = cohere.Client(settings.cohere_api_key)
        return self._cohere

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Rerank retrieved documents. Returns top_k most relevant docs.
        Tries Cohere first, falls back to local cross-encoder.
        """
        top_k = top_k or settings.top_k_rerank
        if len(documents) <= top_k:
            return documents

        try:
            return self._cohere_rerank(query, documents, top_k)
        except Exception as e:
            logger.warning(f"Cohere rerank failed ({e}), falling back to local")
            return self._local_rerank(query, documents, top_k)

    def _cohere_rerank(
        self, query: str, documents: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        client = self._get_cohere()
        response = client.rerank(
            query=query,
            documents=[doc.text for doc in documents],
            model=settings.cohere_rerank_model,
            top_n=top_k,
        )
        reranked = []
        for i, result in enumerate(response.results):
            doc = documents[result.index]
            doc.rank = i + 1
            doc.similarity_score = result.relevance_score
            reranked.append(doc)
        logger.debug(f"Cohere rerank: {len(documents)} → {len(reranked)} docs")
        return reranked

    def _local_rerank(
        self, query: str, documents: list[RetrievedDocument], top_k: int
    ) -> list[RetrievedDocument]:
        """Cross-encoder reranking — slower but zero API cost."""
        if self._local_model is None:
            from sentence_transformers import CrossEncoder
            self._local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, doc.text) for doc in documents]
        scores = self._local_model.predict(pairs)
        ranked = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        for i, (doc, score) in enumerate(ranked):
            doc.rank = i + 1
            doc.similarity_score = float(score)

        return [doc for doc, _ in ranked]
