"""Prepare a deterministic, human-review worksheet from frozen baseline artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "evaluations" / "golden_dataset.csv"
BASELINE_PATH = PROJECT_ROOT / "evaluations" / "results" / "baseline_results.csv"
TRACES_PATH = PROJECT_ROOT / "evaluations" / "results" / "baseline_traces.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluations" / "results" / "human_review_prepared.csv"
REPORT_PATH = PROJECT_ROOT / "evaluations" / "reports" / "review_guide.md"

JUDGMENT_FIELDS = [
    "answer_correctness",
    "citation_support",
    "groundedness",
    "failure_category",
    "reviewer_notes",
]

ABSTENTION_TERMS = (
    "abstain", "do not approve", "do not assume", "do not assert",
    "do not calculate", "do not fabricate", "do not invent", "not established",
    "not enough", "ambiguous", "request clarification", "human review",
)
ACTUAL_ABSTENTION_TERMS = (
    "cannot provide", "do not have enough", "not enough information",
    "cannot determine", "unable to", "ambiguous", "not established",
    "not specified", "unavailable", "insufficient",
)
COVERAGE_TERMS = (
    "rental", "collision", "bodily injury", "property damage", "comprehensive",
    "non-owned", "borrowed vehicle", "endorsement", "deductible", "liability",
    "theft", "wear and tear", "approval", "denied", "covered", "coverage",
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d[\d,]*(?:\.\d+)?)")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json(value: str, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def source_map(trace: dict) -> dict[str, dict]:
    mapping = trace.get("ordered_source_mapping", {})
    if isinstance(mapping, list):
        return {item.get("source_label", ""): item for item in mapping if isinstance(item, dict)}
    return mapping if isinstance(mapping, dict) else {}


def citations_for(trace: dict, baseline: dict) -> list[dict]:
    citations = trace.get("citation_validation", {}).get("accepted_citations")
    if not isinstance(citations, list):
        citations = parse_json(baseline.get("validated_citations", ""), [])
    return [item for item in citations if isinstance(item, dict)]


def evidence_and_filenames(trace: dict, citations: list[dict]) -> tuple[str, str, list[str]]:
    mapping = source_map(trace)
    excerpts: list[str] = []
    filenames: list[str] = []
    for citation in citations:
        label = citation.get("source_label", "")
        source = mapping.get(label, citation)
        filename = source.get("source_filename", citation.get("source_filename", ""))
        section = source.get("section_heading", citation.get("section_heading", ""))
        text = source.get("text", citation.get("text", ""))
        if filename and filename not in filenames:
            filenames.append(filename)
        if label and text:
            excerpt = f"{label} | {filename} | {section}: {text[:1200]}"
            excerpts.append(excerpt)
    return "; ".join(filenames), "\n\n".join(excerpts) or "NO_VALIDATED_CITATION", filenames


def deterministic_flags(golden: dict, baseline: dict, trace: dict, citations: list[dict]) -> list[str]:
    flags: list[str] = []
    answer = (baseline.get("generated_answer") or "").strip()
    expected = (golden.get("Verified expected answer or expected abstention") or "").strip()
    retrieved_ids = set(parse_json(baseline.get("retrieved_chunk_ids", ""), []))
    citation_errors = trace.get("citation_validation", {}).get("errors", [])
    invalid_ids = [c.get("chunk_id") for c in citations if c.get("chunk_id") not in retrieved_ids]
    if invalid_ids or citation_errors or baseline.get("citation_ids_ok") == "false":
        flags.append("unsupported_citations")
    if answer and (not citations or baseline.get("missing_citations") == "true"):
        flags.append("missing_citations")

    expected_abstention = any(term in expected.casefold() for term in ABSTENTION_TERMS)
    actual_abstention = any(term in answer.casefold() for term in ACTUAL_ABSTENTION_TERMS)
    if expected_abstention and answer and not actual_abstention:
        flags.append("missing_abstention")

    question = golden.get("Question", "").casefold()
    expected_lower = expected.casefold()
    actual_lower = answer.casefold()
    actual_coverage = {term for term in COVERAGE_TERMS if term in actual_lower}
    context_terms = question + " " + expected_lower
    extra_terms = {term for term in actual_coverage if term not in context_terms}
    if extra_terms:
        flags.append("potential_out_of_scope_statement:" + ",".join(sorted(extra_terms)))

    cited_text = " ".join(
        (source_map(trace).get(c.get("source_label"), c).get("text", ""))
        for c in citations
    ).casefold()
    unsupported_terms = sorted(
        term for term in actual_coverage
        if term not in cited_text and term not in {"coverage"}
    )
    if unsupported_terms or (answer and not citations):
        flags.append("potential_unsupported_coverage" + (":" + ",".join(unsupported_terms) if unsupported_terms else ""))

    cited_sources = [source_map(trace).get(c.get("source_label"), c) for c in citations]
    values_by_source = [set(NUMBER_RE.findall(str(source.get("text", "")))) for source in cited_sources]
    distinct_values = set().union(*values_by_source) if values_by_source else set()
    if len(values_by_source) > 1 and len(distinct_values) > 1 and len(NUMBER_RE.findall(answer)) <= 1:
        flags.append("potential_conflicting_sources")
    if ("denied" in cited_text or "not cover" in cited_text) and ("covered" in cited_text or "will pay" in cited_text):
        flags.append("potential_conflicting_sources")
    return list(dict.fromkeys(flags))


def main() -> None:
    golden_rows = load_csv(GOLDEN_PATH)
    baseline_rows = load_csv(BASELINE_PATH)
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    trace_by_id = {row["question_id"]: row for row in traces}
    assert len(golden_rows) == 40, f"Expected 40 golden rows, found {len(golden_rows)}"
    assert set(baseline_by_id) == {row["Question ID"] for row in golden_rows}
    assert set(trace_by_id) == set(baseline_by_id)

    fieldnames = [
        "question_id", "policy_id", "scenario", "question", "expected_answer", "actual_answer",
        "cited_filenames", "evidence_excerpts", "missing_information", "verification_status",
        "deterministic_flags", *JUDGMENT_FIELDS,
    ]
    output_rows: list[dict[str, str]] = []
    flag_ids: defaultdict[str, list[str]] = defaultdict(list)
    for golden in golden_rows:
        question_id = golden["Question ID"]
        baseline = baseline_by_id[question_id]
        trace = trace_by_id[question_id]
        citations = citations_for(trace, baseline)
        filenames, excerpts, _ = evidence_and_filenames(trace, citations)
        flags = deterministic_flags(golden, baseline, trace, citations)
        for flag in flags:
            flag_ids[flag.split(":", 1)[0]].append(question_id)
        output_rows.append({
            "question_id": question_id,
            "policy_id": golden["Policy ID"],
            "scenario": golden["Scenario category"],
            "question": golden["Question"],
            "expected_answer": golden["Verified expected answer or expected abstention"],
            "actual_answer": baseline.get("generated_answer", ""),
            "cited_filenames": filenames,
            "evidence_excerpts": excerpts,
            "missing_information": "; ".join(parse_json(baseline.get("missing_information", ""), [])),
            "verification_status": golden["Verification status"],
            "deterministic_flags": "; ".join(flags) if flags else "NONE_OBSERVED",
            **{field: "UNREVIEWED" for field in JUDGMENT_FIELDS},
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    status_counts = Counter(row["verification_status"] for row in output_rows)
    priority_ids = ["AC-021", "AC-032", "AC-033"]
    flagged_non_review = [row for row in output_rows if row["deterministic_flags"] != "NONE_OBSERVED"]
    report = [
        "# Human Review Guide",
        "",
        "This worksheet was prepared from the frozen golden dataset, baseline results, baseline traces, and existing human-review template. No model judge, semantic string comparison, or human judgment was run.",
        "",
        f"- Worksheet rows: {len(output_rows)}",
        f"- VERIFIED: {status_counts.get('VERIFIED', 0)}",
        f"- NEEDS_REVIEW: {status_counts.get('NEEDS_REVIEW', 0)}",
        f"- Deterministically flagged rows: {len(flagged_non_review)}",
        "- All human judgment fields are `UNREVIEWED`.",
        "",
        "## Review priority",
        "",
        "Priority 1 is the three golden cases already marked NEEDS_REVIEW: AC-021, AC-032, and AC-033. Their expected answers were not reclassified here. Review whether the source evidence justifies the expected abstention or answer before using them for evaluation conclusions.",
        "",
        "Priority 2 is every row with a value other than `NONE_OBSERVED` in `deterministic_flags`. These are screening indicators for human inspection, not correctness judgments.",
        "",
        "### Deterministic screening summary",
    ]
    if flag_ids:
        for flag, ids in sorted(flag_ids.items()):
            report.append(f"- `{flag}`: {len(ids)} row(s): {', '.join(ids)}")
    else:
        report.append("- No deterministic screening flags were observed.")
    report.extend([
        "",
        "## Reviewer instructions",
        "",
        "1. Compare the actual answer with the expected answer and the cited source excerpt; record `PASS`, `FAIL`, or `UNCERTAIN` in `answer_correctness`.",
        "2. Record whether each cited filename, section, and excerpt supports the material claim in `citation_support`.",
        "3. Record whether the answer is grounded in the cited evidence, including any extra claim, unsupported coverage statement, or missing abstention, in `groundedness`.",
        "4. Choose a failure category only after inspecting the evidence. Use reviewer notes for the exact source text or reasoning.",
        "5. Treat `unsupported_citations`, `missing_citations`, `missing_abstention`, `potential_out_of_scope_statement`, `potential_unsupported_coverage`, and `potential_conflicting_sources` as prompts for inspection. They do not establish that the answer is wrong.",
        "",
        "The worksheet preserves question IDs, policy IDs, verification statuses, expected answers, generated answers, citation filenames, and evidence excerpts. The original baseline artifacts and RAG pipeline were not changed.",
    ])
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output_rows), "status_counts": status_counts, "flag_ids": flag_ids}, indent=2, default=list))


if __name__ == "__main__":
    main()
