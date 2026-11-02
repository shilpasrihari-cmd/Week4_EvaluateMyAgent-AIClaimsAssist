from types import SimpleNamespace

from evaluations.diagnostic_tracing import (
    DiagnosticContext,
    TracedSemanticRetriever,
    redact_sensitive,
)
from generation.rag_pipeline import RAGPipeline
from retrieval.reranker import RerankedResult


def _passage() -> RerankedResult:
    return RerankedResult(
        chunk_id="chunk-1",
        policy_id="MOT-0000002",
        source_filename="policy.pdf",
        section_heading="Limits",
        text="The limit is $100,000.",
        original_rrf_score=0.03,
        cross_encoder_score=0.9,
        original_rank=1,
        reranked_rank=1,
        retrieval_source="hybrid",
    )


class _Retriever:
    def search(self, query, **kwargs):
        return ["candidate"]


class _Reranker:
    def rerank(self, query, candidates, **kwargs):
        return [_passage()]


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="The limit is $100,000. [Source 1]"))]
        )


def test_trace_captures_raw_output_source_mapping_and_citation_validation():
    trace = {}
    result = RAGPipeline(_Retriever(), _Reranker(), client=_Client()).answer(
        "What is the limit?", policy_id="MOT-0000002", trace=trace
    )

    assert result["citations"][0]["source_label"] == "Source 1"
    assert trace["raw_llm_output"].endswith("[Source 1]")
    assert trace["ordered_source_mapping"]["Source 1"]["chunk_id"] == "chunk-1"
    assert trace["citation_validation"]["accepted_labels"] == ["Source 1"]
    assert trace["fallback_status"] == "none"


def test_redaction_removes_credentials_from_nested_trace_values():
    value = {
        "api_key": "secret-value",
        "nested": "Authorization: Bearer bearer-secret token=token-secret",
    }

    redacted = redact_sensitive(value)

    assert "secret-value" not in str(redacted)
    assert "bearer-secret" not in str(redacted)
    assert "token-secret" not in str(redacted)
    assert "<redacted>" in str(redacted)


def test_semantic_trace_records_errors_without_exposing_exception_text():
    class FailingRetriever:
        def search(self, query, **kwargs):
            raise RuntimeError("request token=secret-value failed")

    context = DiagnosticContext()
    trace = {}
    context.begin(trace)
    wrapped = TracedSemanticRetriever(FailingRetriever(), context)

    try:
        wrapped.search("query")
    except RuntimeError:
        pass

    assert trace["semantic_retrieval"]["count"] == 0
    assert trace["semantic_retrieval"]["error_type"] == "RuntimeError"
    assert "secret-value" not in trace["semantic_retrieval"]["error"]
    assert "secret-value" not in str(trace)
