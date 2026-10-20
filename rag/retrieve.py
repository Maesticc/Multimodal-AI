"""Retrieval: dense (ChromaDB cosine), sparse (BM25) and hybrid (Reciprocal Rank Fusion).

Hybrid retrieval runs both retrievers, takes the top `candidate_k` from each
and fuses them with RRF:  score(d) = sum_r 1 / (rrf_k + rank_r(d)).
RRF only uses ranks, so the incomparable scales of cosine similarity and BM25
do not need to be calibrated against each other.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rank_bm25 import BM25Okapi

from .config import Settings
from .index import embed_query, get_collection, load_chunks
from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_BM25_STOP = {
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "is", "are", "was", "were", "be", "by",
    "with", "as", "at", "that", "this", "it", "its", "from", "what", "which", "how", "does", "do", "why",
    "who", "when", "we", "they", "their", "paper", "authors",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _BM25_STOP]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rank: int  # final 1-based rank
    score: float  # score used for the final ranking (cosine sim, BM25 or RRF)
    dense_rank: int | None = None
    dense_score: float | None = None  # cosine similarity
    bm25_rank: int | None = None
    bm25_score: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self, max_chars: int | None = None) -> dict:
        d = asdict(self)
        if max_chars is not None and len(d["chunk"]["text"]) > max_chars:
            d["chunk"]["text"] = d["chunk"]["text"][:max_chars] + " ..."
        return d


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks = load_chunks(settings)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.section} {c.text}") for c in self.chunks])
        self.collection = get_collection(settings)

    # -- individual retrievers ------------------------------------------------
    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        res = self.collection.query(query_embeddings=[embed_query(self.settings, query)], n_results=k)
        # Chroma returns cosine *distance*; convert to similarity.
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]

    # -- public API -------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None, mode: str | None = None) -> list[RetrievedChunk]:
        s = self.settings
        k = top_k or s.top_k
        mode = (mode or s.retrieval_mode).lower()
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        cand = max(k, s.candidate_k)
        dense = self.dense(query, cand) if mode in {"dense", "hybrid"} else []
        sparse = self.sparse(query, cand) if mode in {"bm25", "hybrid"} else []
        d_rank = {cid: (r, sc) for r, (cid, sc) in enumerate(dense, start=1)}
        s_rank = {cid: (r, sc) for r, (cid, sc) in enumerate(sparse, start=1)}

        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        else:
            fused: dict[str, float] = {}
            for lst in (dense, sparse):
                for r, (cid, _) in enumerate(lst, start=1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (s.rrf_k + r)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        out = []
        for r, (cid, sc) in enumerate(ranked[:k], start=1):
            dr, ds = d_rank.get(cid, (None, None))
            br, bs = s_rank.get(cid, (None, None))
            out.append(
                RetrievedChunk(
                    chunk=self.by_id[cid], rank=r, score=float(sc),
                    dense_rank=dr, dense_score=ds, bm25_rank=br, bm25_score=bs,
                )
            )
        return out
