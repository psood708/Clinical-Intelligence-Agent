"""
HNSW vector index for clinical document retrieval.
This is where your existing HNSW project connects — it's the retrieval
backbone of the entire agent system.

Usage:
    index = ClinicalHNSWIndex()
    index.build(documents, embeddings)
    results = index.search(query_embedding, k=20)
"""
import os
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from src.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    metadata: dict
    similarity_score: float
    rank: int


@dataclass
class HNSWNode:
    """Graph node — each clinical document is one node."""
    doc_id: str
    embedding: np.ndarray
    text: str
    metadata: dict = field(default_factory=dict)
    # Adjacency lists per layer: {layer_idx: [neighbor_node_ids]}
    neighbors: dict[int, list[str]] = field(default_factory=dict)


class ClinicalHNSWIndex:
    """
    HNSW index built from scratch (your existing project).
    M=16, ef_construction=200 — tuned for clinical text recall vs. latency.

    Key properties:
    - 98% recall on 1M-vector benchmarks (from your project README)
    - Sub-10ms p95 latency at 1M vectors
    - No managed vector DB dependency
    """

    def __init__(
        self,
        M: int | None = None,
        ef_construction: int | None = None,
        ef_search: int | None = None,
    ):
        self.M = M or settings.hnsw_m
        self.ef_construction = ef_construction or settings.hnsw_ef_construction
        self.ef_search = ef_search or settings.hnsw_ef_search
        self.nodes: dict[str, HNSWNode] = {}
        self.entry_point: str | None = None
        self.max_layer: int = 0
        self._index_path = Path(settings.hnsw_index_path)

    # ─── Build ─────────────────────────────────────────────────────────────

    def build(self, documents: list[dict], embeddings: np.ndarray) -> None:
        """
        Build index from a list of document dicts and their embeddings.
        documents: [{"doc_id": str, "text": str, "metadata": dict}, ...]
        embeddings: (N, 768) float32 numpy array
        """
        assert len(documents) == len(embeddings), "Mismatched documents / embeddings"
        logger.info(f"Building HNSW index over {len(documents)} documents")

        for doc, emb in zip(documents, embeddings):
            self._insert(doc["doc_id"], emb, doc["text"], doc.get("metadata", {}))

        logger.info(f"Index built — {len(self.nodes)} nodes, max_layer={self.max_layer}")

    def _insert(self, doc_id: str, embedding: np.ndarray, text: str, metadata: dict):
        """Insert a single node into the HNSW graph."""
        level = self._random_level()
        node = HNSWNode(
            doc_id=doc_id,
            embedding=embedding,
            text=text,
            metadata=metadata,
        )
        self.nodes[doc_id] = node

        if self.entry_point is None:
            self.entry_point = doc_id
            self.max_layer = level
            return

        # Greedy search from top layer down to level+1
        ep = self.entry_point
        for lc in range(self.max_layer, level, -1):
            ep = self._greedy_search(embedding, ep, lc)

        # Connect at each layer from level down to 0
        for lc in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(embedding, ep, self.ef_construction, lc)
            neighbors = self._select_neighbors(candidates, self.M)
            node.neighbors[lc] = [n for n, _ in neighbors]

            for neighbor_id, _ in neighbors:
                neighbor = self.nodes[neighbor_id]
                if lc not in neighbor.neighbors:
                    neighbor.neighbors[lc] = []
                neighbor.neighbors[lc].append(doc_id)
                if len(neighbor.neighbors[lc]) > 2 * self.M:
                    neighbor.neighbors[lc] = neighbor.neighbors[lc][: 2 * self.M]

            if candidates:
                ep = candidates[0][0]

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = doc_id

    # ─── Search ────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedDocument]:
        """Return top-k most similar documents."""
        if self.entry_point is None:
            return []

        ep = self.entry_point
        for lc in range(self.max_layer, 0, -1):
            ep = self._greedy_search(query_embedding, ep, lc)

        candidates = self._search_layer(query_embedding, ep, max(self.ef_search, k), 0)
        top_k = candidates[:k]

        return [
            RetrievedDocument(
                doc_id=doc_id,
                text=self.nodes[doc_id].text,
                metadata=self.nodes[doc_id].metadata,
                similarity_score=float(score),
                rank=rank + 1,
            )
            for rank, (doc_id, score) in enumerate(top_k)
        ]

    # ─── Internals ─────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _random_level(self) -> int:
        import math, random
        return int(-math.log(random.random()) * (1 / math.log(self.M)))

    def _greedy_search(self, query: np.ndarray, ep_id: str, layer: int) -> str:
        best = ep_id
        best_sim = self._cosine_similarity(query, self.nodes[ep_id].embedding)
        changed = True
        while changed:
            changed = False
            for nb_id in self.nodes[best].neighbors.get(layer, []):
                sim = self._cosine_similarity(query, self.nodes[nb_id].embedding)
                if sim > best_sim:
                    best_sim, best, changed = sim, nb_id, True
        return best

    def _search_layer(
        self, query: np.ndarray, ep_id: str, ef: int, layer: int
    ) -> list[tuple[str, float]]:
        import heapq
        ep_sim = self._cosine_similarity(query, self.nodes[ep_id].embedding)
        candidates = [(-ep_sim, ep_id)]
        dynamic_list = [(ep_sim, ep_id)]
        visited = {ep_id}

        while candidates:
            neg_sim, c_id = heapq.heappop(candidates)
            c_sim = -neg_sim
            worst_in_list = dynamic_list[0][0] if dynamic_list else 0
            if c_sim < worst_in_list:
                break
            for nb_id in self.nodes[c_id].neighbors.get(layer, []):
                if nb_id not in visited:
                    visited.add(nb_id)
                    nb_sim = self._cosine_similarity(query, self.nodes[nb_id].embedding)
                    if len(dynamic_list) < ef or nb_sim > dynamic_list[0][0]:
                        heapq.heappush(candidates, (-nb_sim, nb_id))
                        heapq.heappush(dynamic_list, (nb_sim, nb_id))
                        if len(dynamic_list) > ef:
                            heapq.heappop(dynamic_list)

        return sorted(dynamic_list, reverse=True)

    def _select_neighbors(
        self, candidates: list[tuple[str, float]], M: int
    ) -> list[tuple[str, float]]:
        return candidates[:M]

    # ─── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        path = path or self._index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "ClinicalHNSWIndex":
        path = path or Path(settings.hnsw_index_path)
        with open(path, "rb") as f:
            return pickle.load(f)
