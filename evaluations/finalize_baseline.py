"""Finalize deterministic Week 4 baseline artifacts after a completed run."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from evaluations.run_baseline import GOLDEN_DATASET, PROJECT_ROOT, load_project_environment


RESULTS_DIR = PROJECT_ROOT / "evaluations" / "results"
RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
TRACES_JSON = RESULTS_DIR / "baseline_traces.json"
METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"
HUMAN_REVIEW_CSV = RESULTS_DIR / "human_review.csv"
REPORT = PROJECT_ROOT / "evaluations" / "reports" / "baseline_report.md"


def _read_rows() -> list[dict[str, str]]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_traces() -> list[dict[str, Any]]:
    return json.loads(TRACES_JSON.read_text(encoding="utf-8"))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _expected_abstention(text: str) -> bool:
    lowered = text.lower()
    indicators = (
        "abstain",
        "do not approve",
        "do not assume",
        "do not assert",
        "do not calculate",
        "do not fabricate",
        "do not invent",
        "not established",
        "not enough",
        "ambiguous",
        "request clarification",
        "human review",
    )
    return any(indicator in lowered for indicator in indicators)


def _explicit_abstention(text: str) -> bool:
    lowered = text.lower()
    indicators = (
        "cannot provide",
        "do not have enough",
        "not enough information",
        "cannot determine",
        "unable to",
        "ambiguous",
        "not established",
        "not specified",
        "unavailable",
        "insufficient",
    )
    return any(indicator in lowered for indicator in indicators)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999) - 1)))
    return ordered[index]


def _git_state() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run_git("status", "--porcelain")
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(status),
        }
    except Exception as exc:
        return {"error_type": type(exc).__name__}


def _configuration() -> dict[str, Any]:
    load_project_environment()
    config: dict[str, Any] = {
        "environment_file": str(PROJECT_ROOT / ".env"),
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "aiclaimsassist"),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "pinecone_api_key_present": bool(os.getenv("PINECONE_API_KEY")),
        "nebius_api_key_present": bool(os.getenv("NEBIUS_API_KEY")),
        "retrieval_mode": "hybrid",
        "chunking_strategy": "section_aware",
        "rrf_k": 60,
        "retrieval_top_k": 10,
        "passage_top_k": 5,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "llm_model": "google/gemma-3-27b-it",
        "policy_filtering": "explicit policy_id per question",
    }
    try:
        import truststore

        truststore.inject_into_ssl()
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY missing")
        client = Pinecone(api_key=api_key)
        name = config["pinecone_index"]
        description = client.describe_index(name)
        index = client.Index(name)
        stats = index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", {}) or {}
        config["pinecone"] = {
            "name": getattr(description, "name", name),
            "dimension": getattr(description, "dimension", None),
            "metric": getattr(description, "metric", None),
            "ready": str(getattr(description, "status", None)),
            "namespace_vector_counts": {
                str(namespace): getattr(value, "vector_count", None)
                if not isinstance(value, dict)
                else value.get("vector_count")
                for namespace, value in namespaces.items()
            },
        }
    except Exception as exc:
        config["pinecone_error_type"] = type(exc).__name__
    config["git"] = _git_state()
    return config


def _human_review(rows: list[dict[str, str]]) -> None:
    HUMAN_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "policy_id",
        "question",
        "scenario",
        "verification_status",
        "expected_answer",
        "generated_answer",
        "validated_citations",
        "retrieved_chunk_ids",
        "answer_correctness",
        "citation_support",
        "groundedness",
        "failure_category",
        "notes",
    ]
    with HUMAN_REVIEW_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "policy_id": row["policy_id"],
                    "question": row["question"],
                    "scenario": row["scenario"],
                    "verification_status": row["verification_status"],
                    "expected_answer": row["expected_answer"],
                    "generated_answer": row["generated_answer"],
                    "validated_citations": row["validated_citations"],
                    "retrieved_chunk_ids": row["retrieved_chunk_ids"],
                    "answer_correctness": "UNREVIEWED",
                    "citation_support": "UNREVIEWED",
                    "groundedness": "UNREVIEWED",
                    "failure_category": "UNREVIEWED",
                    "notes": "UNREVIEWED",
                }
            )


def finalize() -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = _read_rows()
    traces = _read_traces()
    if len(rows) != len(traces):
        raise ValueError("baseline results and traces have different lengths")

    total = len(rows)
    verified = [row for row in rows if row["verification_status"] == "VERIFIED"]
    needs_review = [row for row in rows if row["verification_status"] == "NEEDS_REVIEW"]
    execution_success = [
        row for row in rows if row["api_failure"] == "false" and row["empty_response"] == "false"
    ]
    citation_entries = []
    valid_citation_entries = []
    missing_citation_cases = []
    error_counts: Counter[str] = Counter()
    for row in rows:
        citations = json.loads(row["validated_citations"] or "[]")
        sources = json.loads(row["retrieved_chunk_ids"] or "[]")
        for citation in citations:
            citation_entries.append(citation)
            if citation.get("chunk_id") in sources and str(citation.get("source_label", "")).startswith("Source "):
                valid_citation_entries.append(citation)
        if row["missing_citations"] == "true":
            missing_citation_cases.append(row["question_id"])
        error_counts.update(json.loads(row["errors"] or "[]"))

    answerable = [row for row in rows if row["generated_answer"].strip()]
    expected_abstentions = [
        row for row in verified if _expected_abstention(row["expected_answer"])
    ]
    explicit_abstentions = [
        row for row in expected_abstentions if _explicit_abstention(row["generated_answer"])
    ]
    latencies = [float(row["latency_seconds"]) for row in rows]
    metrics: dict[str, Any] = {
        "dataset": {
            "total": total,
            "verified": len(verified),
            "needs_review": len(needs_review),
            "question_ids": [row["question_id"] for row in rows],
        },
        "execution_success": {
            "numerator": len(execution_success),
            "denominator": total,
            "rate": len(execution_success) / total if total else None,
        },
        "policy_isolation": {
            "numerator": sum(row["policy_isolation_ok"] == "true" for row in rows),
            "denominator": total,
            "rate": sum(row["policy_isolation_ok"] == "true" for row in rows) / total if total else None,
        },
        "citation_reference_validity": {
            "numerator": len(valid_citation_entries),
            "denominator": len(citation_entries),
            "rate": len(valid_citation_entries) / len(citation_entries) if citation_entries else None,
        },
        "citation_coverage": {
            "numerator": sum(bool(json.loads(row["validated_citations"] or "[]")) for row in answerable),
            "denominator": len(answerable),
            "rate": sum(bool(json.loads(row["validated_citations"] or "[]")) for row in answerable) / len(answerable)
            if answerable
            else None,
        },
        "verified_abstention": {
            "numerator": len(explicit_abstentions),
            "denominator": len(expected_abstentions),
            "rate": len(explicit_abstentions) / len(expected_abstentions)
            if expected_abstentions
            else None,
            "interpretation": "explicit abstention indicator only; not a semantic correctness judgment",
        },
        "recall_at_5": {
            "numerator": 0,
            "denominator": 0,
            "rate": None,
            "reason": "No verified relevant chunk-ID labels are present in the golden dataset.",
        },
        "latency_seconds": {
            "numerator": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
        "error_rate": {
            "numerator": sum(bool(json.loads(row["errors"] or "[]")) for row in rows),
            "denominator": total,
            "rate": sum(bool(json.loads(row["errors"] or "[]")) for row in rows) / total if total else None,
            "by_error": dict(error_counts),
        },
        "missing_citations": {
            "numerator": len(missing_citation_cases),
            "denominator": total,
            "question_ids": missing_citation_cases,
        },
        "review_cases": [row["question_id"] for row in needs_review],
        "configuration": _configuration(),
    }
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _human_review(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Week 4 Baseline Evaluation Report",
        "",
        f"The run evaluated {total} questions sequentially: {len(verified)} VERIFIED and {len(needs_review)} NEEDS_REVIEW.",
        "Semantic answer correctness, citation support, and groundedness are left UNREVIEWED in the human-review queue.",
        "",
        "## Frozen configuration",
        "",
        f"- Git commit: `{metrics['configuration']['git'].get('commit', 'UNAVAILABLE')}`; working tree dirty at run time: `{metrics['configuration']['git'].get('working_tree_dirty', 'UNKNOWN')}`.",
        f"- Pinecone index: `{metrics['configuration']['pinecone_index']}`.",
        f"- Embedding model: `{metrics['configuration']['embedding_model']}`.",
        "- Namespaces: `section-aware` and `fixed-size`.",
        "- Retrieval: hybrid, section-aware, RRF k=60, up to 10 candidates, top 5 passages, CrossEncoder reranking.",
        f"- Reranker: `{metrics['configuration']['reranker_model']}`.",
        f"- LLM: `{metrics['configuration']['llm_model']}`.",
        "",
        "## Deterministic metrics",
        "",
        f"- Execution success: {metrics['execution_success']['numerator']}/{metrics['execution_success']['denominator']} ({metrics['execution_success']['rate']:.4f}).",
        f"- Policy isolation: {metrics['policy_isolation']['numerator']}/{metrics['policy_isolation']['denominator']} ({metrics['policy_isolation']['rate']:.4f}).",
        f"- Citation reference validity: {metrics['citation_reference_validity']['numerator']}/{metrics['citation_reference_validity']['denominator']} ({metrics['citation_reference_validity']['rate'] if metrics['citation_reference_validity']['rate'] is not None else 'N/A'}).",
        f"- Citation coverage: {metrics['citation_coverage']['numerator']}/{metrics['citation_coverage']['denominator']} ({metrics['citation_coverage']['rate'] if metrics['citation_coverage']['rate'] is not None else 'N/A'}).",
        f"- Verified abstention indicator: {metrics['verified_abstention']['numerator']}/{metrics['verified_abstention']['denominator']} ({metrics['verified_abstention']['rate'] if metrics['verified_abstention']['rate'] is not None else 'N/A'}).",
        "- Recall@5: N/A (0/0); the golden dataset has no verified relevant chunk-ID labels.",
        f"- Latency seconds: mean {metrics['latency_seconds']['mean']:.6f}, median {metrics['latency_seconds']['median']:.6f}, P95 {metrics['latency_seconds']['p95']:.6f}, denominator {metrics['latency_seconds']['numerator']}.",
        f"- Error rate: {metrics['error_rate']['numerator']}/{metrics['error_rate']['denominator']} ({metrics['error_rate']['rate']:.4f}).",
        f"- Missing-citation cases: {metrics['missing_citations']['numerator']}/{metrics['missing_citations']['denominator']}; IDs: {', '.join(metrics['missing_citations']['question_ids']) or 'none'}.",
        "",
        "## Review queue",
        "",
        f"NEEDS_REVIEW cases: {', '.join(metrics['review_cases']) or 'none'}.",
        "All 40 rows in `human_review.csv` have UNREVIEWED answer correctness, citation support, groundedness, failure category, and notes fields.",
        "",
        "## Errors",
        "",
        _json(metrics["error_rate"]["by_error"]) if metrics["error_rate"]["by_error"] else "No errors were recorded.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics, rows


if __name__ == "__main__":
    finalize()
