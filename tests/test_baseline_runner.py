import csv
import json
from pathlib import Path
from types import SimpleNamespace

from evaluations.run_baseline import deterministic_checks, load_project_environment, run
from retrieval.reranker import RerankedResult


def _passage(chunk_id="chunk-1", policy_id="MOT-0000002"):
    return RerankedResult(
        chunk_id=chunk_id,
        policy_id=policy_id,
        source_filename="MOT-0000002-declarations.pdf",
        section_heading="Limits of Insurance",
        text="Bodily injury per person €100,000.",
        original_rrf_score=0.03,
        cross_encoder_score=0.9,
        original_rank=1,
        reranked_rank=1,
        retrieval_source="hybrid",
    )


class FakePipeline:
    model_name = "google/gemma-3-27b-it"

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "answer": "The limit is €100,000. [Source 1]",
            "citations": [{
                "source_label": "Source 1",
                "chunk_id": "chunk-1",
                "policy_id": "MOT-0000002",
                "source_filename": "MOT-0000002-declarations.pdf",
                "section_heading": "Limits of Insurance",
                "text": "Bodily injury per person €100,000.",
            }],
            "missing_information": [],
            "retrieved_sources": [{
                "chunk_id": "chunk-1",
                "policy_id": "MOT-0000002",
                "source_filename": "MOT-0000002-declarations.pdf",
                "section_heading": "Limits of Insurance",
                "text": "Bodily injury per person €100,000.",
            }],
            "model_name": self.model_name,
            "latency": 0.25,
        }

    def answer(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.result


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "golden.csv"
    path.write_text(
        "Question ID,Policy ID,Question,Scenario category,Verified expected answer or expected abstention,Supporting filename and section,Evidence excerpt,Verification status\n"
        "AC-001,MOT-0000002,What is the limit?,Happy path,€100000,source,excerpt,VERIFIED\n",
        encoding="utf-8",
    )
    return path


def test_runner_passes_explicit_evaluation_configuration_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_CSV", tmp_path / "results" / "baseline_results.csv")
    monkeypatch.setattr("evaluations.run_baseline.TRACES_JSON", tmp_path / "results" / "baseline_traces.json")
    pipeline = FakePipeline()

    rows, traces = run(dataset_path=_dataset(tmp_path), pipeline_factory=lambda: pipeline)

    assert pipeline.calls == [(
        "What is the limit?",
        {
            "policy_id": "MOT-0000002",
            "strategy": "section_aware",
            "retrieval_mode": "hybrid",
            "reranking_enabled": True,
        },
    )]
    assert rows[0]["retrieved_chunk_ids"] == '["chunk-1"]'
    assert rows[0]["policy_isolation_ok"] == "true"
    assert json.loads((tmp_path / "results" / "baseline_traces.json").read_text())[0]["configuration"]["retrieval_top_k"] == 10
    with (tmp_path / "results" / "baseline_results.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_deterministic_checks_flag_isolation_missing_citations_and_api_failure():
    checks = deterministic_checks(
        {
            "answer": "A response",
            "citations": [{"source_label": "Source 1", "chunk_id": "missing", "policy_id": "MOT-0000004"}],
            "retrieved_sources": [{"chunk_id": "actual", "policy_id": "MOT-0000002"}],
            "error": "RuntimeError",
        },
        policy_id="MOT-0000002",
    )

    assert checks["policy_isolation_ok"] is False
    assert checks["citation_ids_ok"] is False
    assert checks["missing_citations"] is False
    assert checks["api_failure"] is True
    assert "policy_isolation_violation" in checks["errors"]
    assert "invalid_citation_id" in checks["errors"]
    assert "api_failure:RuntimeError" in checks["errors"]


def test_verified_only_and_limit_keep_needs_review_separate(tmp_path, monkeypatch):
    path = tmp_path / "golden.csv"
    path.write_text(
        "Question ID,Policy ID,Question,Scenario category,Verified expected answer or expected abstention,Supporting filename and section,Evidence excerpt,Verification status\n"
        "AC-001,MOT-0000002,Q1,Happy path,A,source,excerpt,VERIFIED\n"
        "AC-021,MOT-0000004,Q2,Edge case,B,source,excerpt,NEEDS_REVIEW\n"
        "AC-002,MOT-0000004,Q3,Happy path,C,source,excerpt,VERIFIED\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_CSV", tmp_path / "results" / "baseline_results.csv")
    monkeypatch.setattr("evaluations.run_baseline.TRACES_JSON", tmp_path / "results" / "baseline_traces.json")
    pipeline = FakePipeline()

    rows, _ = run(dataset_path=path, limit=2, verified_only=True, pipeline_factory=lambda: pipeline)

    assert [row["question_id"] for row in rows] == ["AC-001", "AC-002"]
    assert [call[1]["policy_id"] for call in pipeline.calls] == ["MOT-0000002", "MOT-0000004"]


def test_pipeline_setup_failure_is_recorded_per_question(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("evaluations.run_baseline.RESULTS_CSV", tmp_path / "results" / "baseline_results.csv")
    monkeypatch.setattr("evaluations.run_baseline.TRACES_JSON", tmp_path / "results" / "baseline_traces.json")

    rows, traces = run(
        dataset_path=_dataset(tmp_path),
        pipeline_factory=lambda: (_ for _ in ()).throw(RuntimeError("missing credentials")),
    )

    assert rows[0]["generated_answer"] == ""
    assert rows[0]["api_failure"] == "true"
    assert "api_failure:RuntimeError" in json.loads(rows[0]["errors"])
    assert traces[0]["result"]["error"] == "RuntimeError"


def test_project_env_load_is_independent_of_working_directory(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    env_path = project_root / ".env"
    env_path.write_text("BASELINE_TEST_TOKEN=loaded-from-project-root\n", encoding="utf-8")
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.setattr("evaluations.run_baseline.PROJECT_ROOT", project_root)
    monkeypatch.chdir(other_directory)
    monkeypatch.delenv("BASELINE_TEST_TOKEN", raising=False)

    loaded_path = load_project_environment()

    assert loaded_path == env_path
    assert __import__("os").environ["BASELINE_TEST_TOKEN"] == "loaded-from-project-root"
