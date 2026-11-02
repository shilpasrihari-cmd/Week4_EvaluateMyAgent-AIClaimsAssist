"""Semantic retrieval from the existing Pinecone vector index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from ingestion.chunker import chunk_documents
from ingestion.indexer import EXPECTED_DIMENSION, connect_to_index, load_embedding_model


NAMESPACE_BY_STRATEGY = {"fixed_size": "fixed-size", "section_aware": "section-aware"}
STRATEGY_ALIASES = {
    "fixed_size": "fixed_size",
    "fixed-size": "fixed_size",
    "section_aware": "section_aware",
    "section-aware": "section_aware",
}
_POLICY_ID_PATTERN = re.compile(r"\b(?:MOT|COM)-\d+\b", re.IGNORECASE)


def extract_policy_id(text: str) -> str | None:
    match = _POLICY_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


class SemanticRetrievalError(RuntimeError):
    """Raised when the semantic query cannot be executed."""


@dataclass(frozen=True)
class SemanticResult:
    chunk_id: str
    policy_id: str
    source_filename: str
    section_heading: str
    text: str
    semantic_score: float


class SemanticRetriever:
    def __init__(
        self,
        index: Any,
        embedder: Any,
        valid_policy_ids: set[str],
        *,
        dimension: int = EXPECTED_DIMENSION,
        strategy: str | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.valid_policy_ids = frozenset(valid_policy_ids)
        self.dimension = dimension
        self.strategy = self._normalize_strategy(strategy) if strategy else None

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        index: Any | None = None,
        embedder: Any | None = None,
        strategy: str | None = None,
    ) -> "SemanticRetriever":
        chunks = chunk_documents(directory, "fixed_size")
        policy_ids = {chunk.policy_id for chunk in chunks}
        return cls(
            index or connect_to_index(),
            embedder or load_embedding_model(),
            policy_ids,
            strategy=strategy,
        )

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        normalized = STRATEGY_ALIASES.get(strategy)
        if normalized is None:
            raise ValueError("strategy must be 'fixed_size' or 'section_aware'")
        return normalized

    def _embed_query(self, query: str) -> list[float]:
        values = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.asarray(values)
        if vector.shape != (1, self.dimension):
            raise SemanticRetrievalError(
                f"Query embedding shape {vector.shape} does not match (1, {self.dimension})."
            )
        return vector[0].tolist()

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = str(exc)
        message = re.sub(
            r"(?i)(api[_-]?key|token|authorization|secret|password)\s*[:=]\s*[^,\s}]+",
            r"\1=<redacted>",
            message,
        )
        message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", message)
        return f"{type(exc).__name__}: {message}"[:500]

    def diagnostic_snapshot(
        self,
        query: str,
        *,
        strategy: str = "section_aware",
        policy_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Inspect semantic configuration and query behavior without mutating the index."""

        normalized_strategy = self._normalize_strategy(strategy)
        requested_policy_id = (policy_id or extract_policy_id(query) or "").upper() or None
        snapshot: dict[str, Any] = {
            "namespace": NAMESPACE_BY_STRATEGY[normalized_strategy],
            "policy_id": requested_policy_id,
            "policy_id_known_locally": requested_policy_id in self.valid_policy_ids
            if requested_policy_id
            else None,
            "policy_filter": {"policy_id": {"$eq": requested_policy_id}}
            if requested_policy_id
            else None,
            "expected_dimension": self.dimension,
        }
        try:
            vector = self._embed_query(query)
            snapshot["query_embedding_dimension"] = len(vector)
            snapshot["query_embedding_norm"] = float(np.linalg.norm(np.asarray(vector)))
        except Exception as exc:
            snapshot["error_type"] = type(exc).__name__
            snapshot["error"] = self._safe_error(exc)
            return snapshot

        snapshot["index_description_available"] = hasattr(self.index, "describe_index")
        if not snapshot["index_description_available"]:
            snapshot["index_description_note"] = (
                "The Pinecone Index data-plane client does not expose describe_index; "
                "index dimension and metric are not observable from this object."
            )

        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
            def vector_count(value: Any) -> Any:
                return (
                    value.get("vector_count")
                    if isinstance(value, dict)
                    else getattr(value, "vector_count", None)
                )

            snapshot["namespace_vector_counts"] = {
                str(name): vector_count(value) for name, value in (namespaces or {}).items()
            }
            snapshot["namespace_vector_count"] = snapshot["namespace_vector_counts"].get(
                snapshot["namespace"]
            )
        except Exception as exc:
            snapshot["describe_index_stats_error"] = self._safe_error(exc)

        try:
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=snapshot["namespace"],
                include_metadata=True,
                filter=snapshot["policy_filter"],
            )
            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            snapshot["query_match_count"] = len(matches)
            snapshot["query_match_policy_ids"] = [
                (match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})).get(
                    "policy_id", ""
                )
                for match in matches
            ]
            snapshot["error_type"] = None
        except Exception as exc:
            snapshot["error_type"] = type(exc).__name__
            snapshot["error"] = self._safe_error(exc)
        return snapshot

    def search(
        self,
        query: str,
        *,
        strategy: str | None = None,
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> list[SemanticResult]:
        if not query.strip() or top_k <= 0:
            return []
        normalized_strategy = self._normalize_strategy(strategy or self.strategy or "section_aware")
        if self.strategy and normalized_strategy != self.strategy:
            raise ValueError("Semantic retriever strategy does not match the requested strategy")
        requested_policy_id = policy_id or extract_policy_id(query)
        if requested_policy_id is not None and requested_policy_id not in self.valid_policy_ids:
            return []

        try:
            response = self.index.query(
                vector=self._embed_query(query),
                top_k=top_k,
                namespace=NAMESPACE_BY_STRATEGY[normalized_strategy],
                include_metadata=True,
                filter={"policy_id": {"$eq": requested_policy_id}} if requested_policy_id else None,
            )
        except Exception as exc:
            raise SemanticRetrievalError(f"Pinecone semantic query failed: {exc}") from exc

        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
        results: list[SemanticResult] = []
        for match in matches:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            result = SemanticResult(
                chunk_id=match.get("id") if isinstance(match, dict) else match.id,
                policy_id=metadata.get("policy_id", ""),
                source_filename=metadata.get("source_filename", ""),
                section_heading=metadata.get("section_heading", ""),
                text=metadata.get("text", ""),
                semantic_score=float(match.get("score", 0.0) if isinstance(match, dict) else match.score),
            )
            if requested_policy_id is None or result.policy_id == requested_policy_id:
                results.append(result)
        return results
