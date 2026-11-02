from pathlib import Path

import numpy as np

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever, canonical_chunk_identity
from retrieval.semantic_retriever import SemanticRetriever


POLICIES = Path(__file__).parents[1] / "data" / "policies"


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        return np.ones((len(sentences), 384), dtype=np.float32)


class FakeIndex:
    def __init__(self, matches=None, error=None):
        self.matches = matches or []
        self.error = error
        self.last_filter = None
        self.last_namespace = None

    def query(self, **kwargs):
        if self.error:
            raise self.error
        self.last_filter = kwargs.get("filter")
        self.last_namespace = kwargs["namespace"]
        return {"matches": self.matches}


def _semantic() -> SemanticRetriever:
    matches = [
        {"id": "shared", "score": 0.9, "metadata": {"policy_id": "MOT-0000002", "source_filename": "a.pdf", "section_heading": "A", "text": "shared text"}},
        {"id": "semantic-only", "score": 0.8, "metadata": {"policy_id": "MOT-0000002", "source_filename": "b.pdf", "section_heading": "B", "text": "semantic text"}},
    ]
    return SemanticRetriever(FakeIndex(matches), FakeEmbedder(), {"MOT-0000002", "MOT-0000004"})


def test_semantic_only_uses_namespace_and_policy_filter() -> None:
    semantic = _semantic()
    result = semantic.search("deductible", strategy="section_aware", policy_id="MOT-0000002")

    assert result[0].chunk_id == "shared"
    assert semantic.index.last_namespace == "section-aware"
    assert semantic.index.last_filter == {"policy_id": {"$eq": "MOT-0000002"}}


def test_hybrid_deduplicates_and_uses_rrf() -> None:
    semantic = _semantic()
    keyword = BM25Retriever({
        "fixed_size": None,
        "section_aware": None,
    })
    keyword._indexes["section_aware"] = type("Index", (), {"search": lambda self, *args, **kwargs: [
        type("Result", (), {"chunk_id": "shared", "policy_id": "MOT-0000002", "source_filename": "a.pdf", "section_heading": "A", "text": "shared text", "score": 4.0})(),
        type("Result", (), {"chunk_id": "keyword-only", "policy_id": "MOT-0000002", "source_filename": "c.pdf", "section_heading": "C", "text": "keyword text", "score": 3.0})(),
    ]})()
    result = HybridRetriever(semantic, keyword).search("query", strategy="section_aware")

    assert [item.chunk_id for item in result] == ["shared", "keyword-only", "semantic-only"]
    assert result[0].retrieval_source == "hybrid"
    assert result[0].score_type == "rrf_score"
    assert result[0].ranking_score == 1 / 61 + 1 / 61


def test_hybrid_merges_verified_equivalent_chunks_and_preserves_original_ids() -> None:
    semantic = _semantic()
    semantic.index.matches = [
        {
            "id": "MOT-0000002-schedule:section_aware:0002:digest",
            "score": 0.9,
            "metadata": {
                "policy_id": "MOT-0000002",
                "source_filename": "MOT-0000002-schedule.pdf",
                "section_heading": "Coverage",
                "text": "Coverage\nLimit: €100,000",
            },
        }
    ]
    keyword = BM25Retriever({"fixed_size": None, "section_aware": None})
    keyword._indexes["section_aware"] = type(
        "Index",
        (),
        {
            "search": lambda self, *args, **kwargs: [
                type(
                    "Result",
                    (),
                    {
                        "chunk_id": "DOC-MOT-0000002-SCHEDULE:section_aware:0002:digest",
                        "policy_id": "MOT-0000002",
                        "source_filename": "MOT-0000002-schedule.pdf",
                        "section_heading": "Coverage",
                        "text": "Coverage\nLimit: €100,000",
                        "score": 4.0,
                    },
                )()
            ],
        },
    )()

    result = HybridRetriever(semantic, keyword).search(
        "query", strategy="section_aware", top_k=5, policy_id="MOT-0000002"
    )

    merged = result[0]
    assert merged.retrieval_source == "hybrid"
    assert merged.ranking_score == 1 / 61 + 1 / 61
    assert merged.source_chunk_ids == (
        "MOT-0000002-schedule:section_aware:0002:digest",
        "DOC-MOT-0000002-SCHEDULE:section_aware:0002:digest",
    )
    assert merged.canonical_id == canonical_chunk_identity(
        "MOT-0000002", "MOT-0000002-schedule.pdf", "Coverage", "Coverage\nLimit: €100,000"
    )


def test_hybrid_does_not_merge_different_boundaries_or_text() -> None:
    semantic = _semantic()
    semantic.index.matches = [
        {
            "id": "semantic-boundary",
            "score": 0.9,
            "metadata": {
                "policy_id": "MOT-0000002",
                "source_filename": "a.pdf",
                "section_heading": "A",
                "text": "shared text plus another sentence",
            },
        }
    ]
    keyword = BM25Retriever({"fixed_size": None, "section_aware": None})
    keyword._indexes["section_aware"] = type(
        "Index", (), {"search": lambda self, *args, **kwargs: [
            type("Result", (), {
                "chunk_id": "keyword-boundary",
                "policy_id": "MOT-0000002",
                "source_filename": "a.pdf",
                "section_heading": "A",
                "text": "shared text",
                "score": 4.0,
            })()
        ]}
    )()

    result = HybridRetriever(semantic, keyword).search("query", strategy="section_aware")

    assert len(result) == 2
    assert {item.retrieval_source for item in result} == {"semantic", "keyword"}


def test_canonical_identity_includes_policy_and_metadata_boundaries() -> None:
    base = canonical_chunk_identity("MOT-0000002", "policy.pdf", "Coverage", "same text")

    assert base != canonical_chunk_identity("MOT-0000004", "policy.pdf", "Coverage", "same text")
    assert base != canonical_chunk_identity("MOT-0000002", "other.pdf", "Coverage", "same text")
    assert base != canonical_chunk_identity("MOT-0000002", "policy.pdf", "Other", "same text")
    assert base != canonical_chunk_identity("MOT-0000002", "policy.pdf", "Coverage", "same text plus")


def test_modes_filter_and_failures_are_graceful() -> None:
    semantic = SemanticRetriever(FakeIndex(error=RuntimeError("offline")), FakeEmbedder(), {"MOT-0000002"})
    keyword = BM25Retriever({"fixed_size": None, "section_aware": None})
    keyword._indexes["section_aware"] = type("Index", (), {"search": lambda self, *args, **kwargs: []})()
    retriever = HybridRetriever(semantic, keyword)

    assert retriever.search("query", mode="semantic_only") == []
    assert retriever.search("query", mode="hybrid") == []
    assert retriever.search("query", policy_id="MOT-9999999") == []
    assert retriever.search("") == []
