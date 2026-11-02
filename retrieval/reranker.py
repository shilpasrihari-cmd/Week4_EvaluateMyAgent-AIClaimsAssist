"""Cross-encoder reranking over existing hybrid retrieval candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from sentence_transformers import CrossEncoder

from .hybrid_retriever import HybridResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_POLICY_ID_PATTERN = re.compile(r"\b(?:MOT|COM)-\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class RerankedResult:
    chunk_id: str
    policy_id: str
    source_filename: str
    section_heading: str
    text: str
    original_rrf_score: float
    cross_encoder_score: float
    original_rank: int
    reranked_rank: int
    retrieval_source: str
    canonical_id: str | None = None
    source_chunk_ids: tuple[str, ...] = ()


class CrossEncoderReranker:
    """Load one CrossEncoder and reuse it for all reranking calls."""

    def __init__(self, model: Any | None = None, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model = model if model is not None else CrossEncoder(model_name)
        self.last_error: str | None = None

    @staticmethod
    def _query_policy_id(query: str) -> str | None:
        match = _POLICY_ID_PATTERN.search(query)
        return match.group(0).upper() if match else None

    def rerank(
        self,
        query: str,
        candidates: Iterable[HybridResult],
        *,
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> list[RerankedResult]:
        self.last_error = None
        if not query.strip() or top_k <= 0:
            return []

        requested_policy_id = policy_id or self._query_policy_id(query)
        unique: dict[str, tuple[int, HybridResult]] = {}
        for original_rank, candidate in enumerate(candidates, start=1):
            if requested_policy_id and candidate.policy_id != requested_policy_id:
                continue
            unique.setdefault(candidate.chunk_id, (original_rank, candidate))
            if len(unique) >= 10:
                break
        selected = list(unique.values())
        if not selected:
            return []

        pairs = [(query, candidate.text) for _, candidate in selected]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            score_values = [float(score) for score in scores]
        except Exception as exc:
            self.last_error = f"Cross-encoder reranking failed: {exc}"
            return []
        if len(score_values) != len(selected):
            self.last_error = "Cross-encoder returned an unexpected number of scores."
            return []

        scored = [
            (original_rank, candidate, score)
            for (original_rank, candidate), score in zip(selected, score_values)
        ]
        scored.sort(key=lambda item: (-item[2], item[0], item[1].chunk_id))
        return [
            RerankedResult(
                chunk_id=candidate.chunk_id,
                policy_id=candidate.policy_id,
                source_filename=candidate.source_filename,
                section_heading=candidate.section_heading,
                text=candidate.text,
                original_rrf_score=candidate.ranking_score,
                cross_encoder_score=score,
                original_rank=original_rank,
                reranked_rank=rank,
                retrieval_source=candidate.retrieval_source,
                canonical_id=candidate.canonical_id,
                source_chunk_ids=candidate.source_chunk_ids or (candidate.chunk_id,),
            )
            for rank, (original_rank, candidate, score) in enumerate(scored[:top_k], start=1)
        ]
