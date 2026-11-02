"""Run the Week 4 baseline evaluation over the validated golden dataset.

The runner deliberately delegates retrieval, reranking, prompting, and citation
validation to the existing :class:`RAGPipeline`. It adds only evaluation
bookkeeping and deterministic integrity checks around the pipeline result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATASET = PROJECT_ROOT / "evaluations" / "golden_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "evaluations" / "results"
RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
TRACES_JSON = RESULTS_DIR / "baseline_traces.json"
DIAGNOSTIC_TRACES_JSON = RESULTS_DIR / "diagnostic_traces.json"
LEGACY_ENV_FILE = PROJECT_ROOT.parents[1] / "week2" / "AIClaimsAssist" / ".env"

REQUIRED_COLUMNS = (
    "Question ID",
    "Policy ID",
    "Question",
    "Scenario category",
    "Verified expected answer or expected abstention",
    "Verification status",
)
CSV_COLUMNS = (
    "question_id",
    "policy_id",
    "question",
    "scenario",
    "verification_status",
    "expected_answer",
    "generated_answer",
    "retrieved_chunk_ids",
    "retrieved_filenames",
    "validated_citations",
    "missing_information",
    "model",
    "latency_seconds",
    "errors",
    "policy_isolation_ok",
    "citation_ids_ok",
    "missing_citations",
    "empty_response",
    "api_failure",
)


def load_questions(path: Path = GOLDEN_DATASET) -> list[dict[str, str]]:
    """Load and validate the golden dataset without changing its contents."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Golden dataset is empty: {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Golden dataset is missing columns: {', '.join(missing)}")
    return rows


def load_project_environment() -> Path | None:
    """Load existing project credentials without depending on the CWD.

    The Week 4 project root is preferred. The repository was cloned from the
    Week 2 project, so its existing local environment is an explicit fallback
    when the Week 4 checkout has no local `.env`. Neither file is created or
    overwritten, and existing process variables take precedence.
    """

    candidates = (PROJECT_ROOT / ".env", LEGACY_ENV_FILE)
    for env_path in candidates:
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)
            return env_path
    return None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_policy_ids(result: dict[str, Any]) -> list[str]:
    sources = result.get("retrieved_sources") or []
    citations = result.get("citations") or []
    return [str(source.get("policy_id", "")) for source in [*sources, *citations]]


def deterministic_checks(
    result: dict[str, Any],
    *,
    policy_id: str,
) -> dict[str, Any]:
    """Check metadata integrity; this does not judge semantic correctness."""

    sources = result.get("retrieved_sources") or []
    citations = result.get("citations") or []
    retrieved_chunk_ids = {str(source.get("chunk_id", "")) for source in sources}
    errors: list[str] = []

    policy_ids = _source_policy_ids(result)
    policy_isolation_ok = all(source_policy_id == policy_id for source_policy_id in policy_ids)
    if not policy_isolation_ok:
        errors.append("policy_isolation_violation")

    citation_ids_ok = True
    for citation in citations:
        chunk_id = str(citation.get("chunk_id", ""))
        source_label = str(citation.get("source_label", ""))
        if not chunk_id or chunk_id not in retrieved_chunk_ids or not source_label.startswith("Source "):
            citation_ids_ok = False
            break
    if not citation_ids_ok:
        errors.append("invalid_citation_id")

    answer = str(result.get("answer") or "")
    empty_response = not answer.strip()
    if empty_response:
        errors.append("empty_response")

    missing_citations = bool(answer.strip()) and not citations
    if missing_citations:
        errors.append("missing_citations")

    api_failure = bool(result.get("error"))
    if api_failure:
        errors.append(f"api_failure:{result['error']}")

    return {
        "errors": errors,
        "policy_isolation_ok": policy_isolation_ok,
        "citation_ids_ok": citation_ids_ok,
        "missing_citations": missing_citations,
        "empty_response": empty_response,
        "api_failure": api_failure,
    }


def evaluate_question(
    question: dict[str, str],
    pipeline: Any,
    diagnostic_context: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one row and return a CSV row plus a detailed trace."""

    question_id = question["Question ID"]
    policy_id = question["Policy ID"].strip().upper()
    query = question["Question"]
    started = time.perf_counter()
    unexpected_error: str | None = None
    trace: dict[str, Any] = {
        "question_id": question_id,
        "policy_id": policy_id,
        "question": query,
        "scenario": question["Scenario category"],
        "verification_status": question["Verification status"],
        "expected_answer": question["Verified expected answer or expected abstention"],
        "configuration": {
            "retrieval_mode": "hybrid",
            "chunking_strategy": "section_aware",
            "reranking_enabled": True,
            "retrieval_top_k": 10,
            "passage_top_k": 5,
        },
    }
    if diagnostic_context is not None:
        diagnostic_context.begin(trace)
    try:
        answer_kwargs = {
            "policy_id": policy_id,
            "strategy": "section_aware",
            "retrieval_mode": "hybrid",
            "reranking_enabled": True,
        }
        if diagnostic_context is not None:
            answer_kwargs["trace"] = trace
        result = pipeline.answer(query, **answer_kwargs)
    except Exception as exc:  # Keep one failed case from stopping the run.
        unexpected_error = type(exc).__name__
        result = {
            "answer": "",
            "citations": [],
            "missing_information": [],
            "retrieved_sources": [],
            "model_name": getattr(pipeline, "model_name", "unknown"),
            "latency": time.perf_counter() - started,
            "error": unexpected_error,
        }

    checks = deterministic_checks(result, policy_id=policy_id)
    if unexpected_error and f"api_failure:{unexpected_error}" not in checks["errors"]:
        checks["errors"].append(f"api_failure:{unexpected_error}")

    sources = result.get("retrieved_sources") or []
    citations = result.get("citations") or []
    latency = float(result.get("latency", time.perf_counter() - started) or 0.0)
    csv_row = {
        "question_id": question_id,
        "policy_id": policy_id,
        "question": query,
        "scenario": question["Scenario category"],
        "verification_status": question["Verification status"],
        "expected_answer": question["Verified expected answer or expected abstention"],
        "generated_answer": result.get("answer", ""),
        "retrieved_chunk_ids": _json([source.get("chunk_id", "") for source in sources]),
        "retrieved_filenames": _json([source.get("source_filename", "") for source in sources]),
        "validated_citations": _json(citations),
        "missing_information": _json(result.get("missing_information") or []),
        "model": result.get("model_name", getattr(pipeline, "model_name", "unknown")),
        "latency_seconds": f"{latency:.6f}",
        "errors": _json(checks["errors"]),
        "policy_isolation_ok": str(checks["policy_isolation_ok"]).lower(),
        "citation_ids_ok": str(checks["citation_ids_ok"]).lower(),
        "missing_citations": str(checks["missing_citations"]).lower(),
        "empty_response": str(checks["empty_response"]).lower(),
        "api_failure": str(checks["api_failure"]).lower(),
    }
    trace["result"] = result
    trace["checks"] = checks
    trace["retrieved_evidence"] = sources
    return csv_row, trace


def write_outputs(
    rows: Iterable[dict[str, Any]],
    traces: list[dict[str, Any]],
    diagnostic_path: Path | None = None,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with TRACES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(traces, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if diagnostic_path is not None:
        from evaluations.diagnostic_tracing import redact_sensitive

        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostic_path.open("w", encoding="utf-8") as handle:
            json.dump(redact_sensitive(traces), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def build_pipeline(*, diagnostic: bool = False, diagnostic_context: Any | None = None) -> Any:
    """Build the existing production pipeline with its existing defaults."""

    load_project_environment()
    from generation.rag_pipeline import RAGPipeline
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import CrossEncoderReranker

    data_root = PROJECT_ROOT / "data" / "sample"
    retriever = HybridRetriever.from_directory(data_root)
    reranker = CrossEncoderReranker()
    if diagnostic:
        from evaluations.diagnostic_tracing import (
            TracedClient,
            TracedHybridRetriever,
            TracedKeywordRetriever,
            TracedReranker,
            TracedSemanticRetriever,
        )

        if diagnostic_context is None:
            raise ValueError("diagnostic_context is required when diagnostic tracing is enabled")
        retriever.semantic = TracedSemanticRetriever(retriever.semantic, diagnostic_context)
        retriever.keyword = TracedKeywordRetriever(retriever.keyword, diagnostic_context)
        pipeline = RAGPipeline(
            TracedHybridRetriever(retriever, diagnostic_context),
            TracedReranker(reranker, diagnostic_context),
        )
        pipeline.client = TracedClient(pipeline.client, diagnostic_context)
        return pipeline
    return RAGPipeline(retriever, reranker)


class _UnavailablePipeline:
    """Adapter that preserves per-question error records after setup failure."""

    model_name = "google/gemma-3-27b-it"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def answer(self, query: str, **kwargs: Any) -> dict[str, Any]:
        raise self.error


def run(
    *,
    dataset_path: Path = GOLDEN_DATASET,
    limit: int | None = None,
    verified_only: bool = False,
    pipeline_factory: Callable[[], Any] = build_pipeline,
    diagnostic: bool = False,
    diagnostic_path: Path = DIAGNOSTIC_TRACES_JSON,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = load_questions(dataset_path)
    if verified_only:
        questions = [question for question in questions if question["Verification status"] == "VERIFIED"]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    from evaluations.diagnostic_tracing import DiagnosticContext

    diagnostic_context = DiagnosticContext() if diagnostic else None
    try:
        if diagnostic and pipeline_factory is build_pipeline:
            pipeline = build_pipeline(diagnostic=True, diagnostic_context=diagnostic_context)
        else:
            pipeline = pipeline_factory()
    except Exception as exc:
        # Keep the selected questions observable in the output files while
        # making the environment/setup failure explicit for every case.
        pipeline = _UnavailablePipeline(exc)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for question in questions:
        row, trace = evaluate_question(question, pipeline, diagnostic_context)
        rows.append(row)
        traces.append(trace)
        # Checkpoint after every question so completed results survive a later
        # case failure or process interruption.
        write_outputs(rows, traces, diagnostic_path if diagnostic else None)
    write_outputs(rows, traces, diagnostic_path if diagnostic else None)
    return rows, traces


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=GOLDEN_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected questions.")
    parser.add_argument("--verified-only", action="store_true", help="Exclude NEEDS_REVIEW questions from this run.")
    parser.add_argument("--diagnostic", action="store_true", help="Capture local retrieval and citation diagnostics.")
    parser.add_argument("--diagnostic-output", type=Path, default=DIAGNOSTIC_TRACES_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, _ = run(
            dataset_path=args.dataset,
            limit=args.limit,
            verified_only=args.verified_only,
            diagnostic=args.diagnostic,
            diagnostic_path=args.diagnostic_output,
        )
    except Exception as exc:
        print(f"Baseline runner failed during setup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} baseline result(s) to {RESULTS_CSV}")
    print(f"Wrote detailed traces to {TRACES_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
