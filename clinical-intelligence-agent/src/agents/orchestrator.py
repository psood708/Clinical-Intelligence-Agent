"""
Orchestrator — LangGraph state machine wiring all four agents.

Flow:
  extract → retrieve → verify → [loop back if low confidence] → synthesize

The loop-back is the key agentic pattern: if the Verifier finds critical
issues, it re-runs extraction on a second pass with explicit correction hints.
Max 2 iterations to prevent infinite loops.
"""
import logging
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator
from src.agents.extractor import ExtractorAgent, ClinicalExtraction
from src.agents.verifier import VerifierAgent, VerificationResult
from src.agents.synthesizer import SynthesizerAgent, ClinicalSummary
from src.retrieval.hnsw_index import ClinicalHNSWIndex, RetrievedDocument
from src.retrieval.embeddings import BioBERTEmbedder
from src.retrieval.reranker import ClinicalReranker
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── State ──────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # Inputs
    document_text: str
    document_id: str

    # Agent outputs
    extraction: ClinicalExtraction | None
    retrieved_docs: list[RetrievedDocument]
    verification: VerificationResult | None
    summary: ClinicalSummary | None

    # Control
    loop_count: int
    should_reextract: bool
    error: str | None


# ─── Graph ──────────────────────────────────────────────────────────────────

class ClinicalPipeline:
    """
    End-to-end clinical document processing pipeline.
    Extract → Retrieve → Verify → (loop) → Synthesize
    """

    def __init__(
        self,
        hnsw_index: ClinicalHNSWIndex,
        embedder: BioBERTEmbedder | None = None,
        reranker: ClinicalReranker | None = None,
    ):
        self.index = hnsw_index
        self.embedder = embedder or BioBERTEmbedder()
        self.reranker = reranker or ClinicalReranker()

        self.extractor = ExtractorAgent()
        self.verifier = VerifierAgent()
        self.synthesizer = SynthesizerAgent()

        self.graph = self._build_graph()

    # ─── Nodes ──────────────────────────────────────────────────────────────

    def _node_extract(self, state: PipelineState) -> dict:
        logger.info(f"[Extract] doc_id={state['document_id']}")
        extraction = self.extractor.extract(state["document_text"])
        return {"extraction": extraction, "should_reextract": False}

    def _node_retrieve(self, state: PipelineState) -> dict:
        logger.info("[Retrieve] HNSW search + Cohere rerank")
        query_emb = self.embedder.embed_single(state["document_text"][:2000])
        candidates = self.index.search(query_emb, k=settings.top_k_retrieval)

        # Rerank — converts top-20 ANN results to top-5 precision results
        query_text = state["extraction"].chief_complaint or state["document_text"][:200]
        reranked = self.reranker.rerank(query_text, candidates, top_k=settings.top_k_rerank)
        logger.info(f"[Retrieve] {len(candidates)} candidates → {len(reranked)} after rerank")
        return {"retrieved_docs": reranked}

    def _node_verify(self, state: PipelineState) -> dict:
        loop = state.get("loop_count", 0) + 1
        logger.info(f"[Verify] iteration {loop}")
        verification = self.verifier.verify(
            original_text=state["document_text"],
            extraction=state["extraction"],
            similar_cases=state["retrieved_docs"],
            iteration=loop,
        )
        critical = [f for f in verification.flags if f.severity == "critical"]
        should_reextract = (
            not verification.is_verified
            and bool(critical)
            and loop < VerifierAgent.MAX_LOOPS
        )
        return {
            "verification": verification,
            "loop_count": loop,
            "should_reextract": should_reextract,
        }

    def _node_synthesize(self, state: PipelineState) -> dict:
        logger.info("[Synthesize] Generating final summary")
        summary = self.synthesizer.synthesize(
            extraction=state["extraction"],
            verification=state["verification"],
            evidence=state["retrieved_docs"],
        )
        return {"summary": summary}

    # ─── Routing ────────────────────────────────────────────────────────────

    def _route_after_verify(self, state: PipelineState) -> str:
        if state.get("should_reextract"):
            logger.info("[Route] Looping back to re-extraction")
            return "extract"
        return "synthesize"

    # ─── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(PipelineState)

        g.add_node("extract", self._node_extract)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("verify", self._node_verify)
        g.add_node("synthesize", self._node_synthesize)

        g.add_edge(START, "extract")
        g.add_edge("extract", "retrieve")
        g.add_edge("retrieve", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"extract": "extract", "synthesize": "synthesize"},
        )
        g.add_edge("synthesize", END)

        return g.compile()

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(self, document_text: str, document_id: str = "doc_0") -> ClinicalSummary:
        """Process a single clinical document end-to-end."""
        initial_state: PipelineState = {
            "document_text": document_text,
            "document_id": document_id,
            "extraction": None,
            "retrieved_docs": [],
            "verification": None,
            "summary": None,
            "loop_count": 0,
            "should_reextract": False,
            "error": None,
        }
        final_state = self.graph.invoke(initial_state)
        if final_state.get("error"):
            raise RuntimeError(final_state["error"])
        return final_state["summary"]
