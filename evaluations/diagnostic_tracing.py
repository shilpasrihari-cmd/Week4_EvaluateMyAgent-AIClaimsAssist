"""Opt-in local tracing adapters for Week 4 evaluation diagnostics."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any


_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|authorization|secret|password)\s*[:=]\s*[^,\s}]+"
)
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")


def sanitize_error(exc: Exception) -> str:
    """Return a bounded error description without credential-like values."""

    message = _BEARER_PATTERN.sub("Bearer <redacted>", str(exc))
    message = _SECRET_KEY_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", message)
    return f"{type(exc).__name__}: {message}"[:500]


def redact_sensitive(value: Any) -> Any:
    """Redact credential-like values in nested trace data."""

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            if re.search(r"(?i)(api[_-]?key|token|authorization|secret|password)", key_text):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub("Bearer <redacted>", value)
        return _SECRET_KEY_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    return value


def _result_summary(result: Any, score_name: str | None = None) -> dict[str, Any]:
    summary = {
        "chunk_id": result.chunk_id,
        "canonical_id": getattr(result, "canonical_id", None),
        "source_chunk_ids": list(getattr(result, "source_chunk_ids", ()) or (result.chunk_id,)),
        "policy_id": result.policy_id,
        "source_filename": result.source_filename,
        "section_heading": result.section_heading,
    }
    if score_name:
        summary[score_name] = getattr(result, score_name, None)
    return summary


class DiagnosticContext:
    """Mutable per-question trace state shared by tracing adapters."""

    def __init__(self) -> None:
        self.current: dict[str, Any] | None = None

    def begin(self, trace: dict[str, Any]) -> None:
        self.current = trace

    def record(self, key: str, value: Any) -> None:
        if self.current is not None:
            self.current[key] = value


class TracedSemanticRetriever:
    def __init__(self, inner: Any, context: DiagnosticContext) -> None:
        self.inner = inner
        self.context = context

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def search(self, query: str, **kwargs: Any) -> list[Any]:
        try:
            snapshot = self.inner.diagnostic_snapshot(query, **kwargs)
        except Exception as exc:
            snapshot = {"error_type": type(exc).__name__, "error": sanitize_error(exc)}
        self.context.record("semantic_index_diagnostic", snapshot)
        try:
            results = self.inner.search(query, **kwargs)
        except Exception as exc:
            self.context.record(
                "semantic_retrieval",
                {"count": 0, "error_type": type(exc).__name__, "error": sanitize_error(exc)},
            )
            raise
        self.context.record(
            "semantic_retrieval",
            {
                "count": len(results),
                "error_type": None,
                "results": [
                    _result_summary(result, "semantic_score") for result in results
                ],
            },
        )
        return results


class TracedKeywordRetriever:
    def __init__(self, inner: Any, context: DiagnosticContext) -> None:
        self.inner = inner
        self.context = context

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def search(self, query: str, **kwargs: Any) -> list[Any]:
        results = self.inner.search(query, **kwargs)
        self.context.record(
            "bm25_retrieval",
            {
                "count": len(results),
                "results": [_result_summary(result, "score") for result in results],
            },
        )
        return results


class TracedHybridRetriever:
    def __init__(self, inner: Any, context: DiagnosticContext) -> None:
        self.inner = inner
        self.context = context

    def search(self, query: str, **kwargs: Any) -> list[Any]:
        try:
            results = self.inner.search(query, **kwargs)
        except Exception as exc:
            self.context.record("hybrid_retrieval", {"count": 0, "error_type": type(exc).__name__})
            raise
        self.context.record(
            "hybrid_retrieval",
            {
                "count": len(results),
                "error_type": None,
                "results": [
                    {
                        **_result_summary(result, "ranking_score"),
                        "retrieval_source": result.retrieval_source,
                    }
                    for result in results
                ],
            },
        )
        return results


class TracedReranker:
    def __init__(self, inner: Any, context: DiagnosticContext) -> None:
        self.inner = inner
        self.context = context

    def rerank(self, query: str, candidates: Any, **kwargs: Any) -> list[Any]:
        candidate_list = list(candidates)
        self.context.record(
            "reranker_input",
            {
                "count": len(candidate_list),
                "results": [_result_summary(result, "ranking_score") for result in candidate_list],
            },
        )
        try:
            results = self.inner.rerank(query, candidate_list, **kwargs)
        except Exception as exc:
            self.context.record("cross_encoder_reranking", {"count": 0, "error_type": type(exc).__name__})
            raise
        self.context.record(
            "cross_encoder_reranking",
            {
                "count": len(results),
                "error_type": getattr(self.inner, "last_error", None),
                "results": [
                    {
                        **_result_summary(result, "cross_encoder_score"),
                        "original_rank": result.original_rank,
                        "reranked_rank": result.reranked_rank,
                    }
                    for result in results
                ],
            },
        )
        return results


class TracedClient:
    """Capture only the model response text; never log request credentials."""

    def __init__(self, inner: Any, context: DiagnosticContext) -> None:
        self.inner = inner
        self.context = context
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        response = self.inner.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        self.context.record("raw_llm_output_from_client", content)
        return response
