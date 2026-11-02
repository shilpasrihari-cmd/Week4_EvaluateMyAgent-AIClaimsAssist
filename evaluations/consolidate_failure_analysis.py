"""Consolidate provisional review signals into a non-confirmatory failure taxonomy."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "evaluations" / "results" / "human_review_prepared.csv"
BATCH_1 = ROOT / "evaluations" / "results" / "provisional_review_ledger.csv"
BATCH_2 = ROOT / "evaluations" / "results" / "review_batch_2.csv"
BASELINE = ROOT / "evaluations" / "results" / "baseline_results.csv"
TRACES = ROOT / "evaluations" / "results" / "baseline_traces.json"
OUT = ROOT / "evaluations" / "results" / "failure_taxonomy.csv"
REPORT = ROOT / "evaluations" / "reports" / "failure_analysis.md"

# Batch 1 contains explicit provisional labels. Batch 2 contains no per-ID labels;
# these three are the failure cases selected from its supplied 7/3 aggregate using
# the deterministic indicators and answer/evidence comparison.
BATCH_1_FAIL = {"AC-003", "AC-023", "AC-027", "AC-029", "AC-031", "AC-033", "AC-034", "AC-035"}
BATCH_2_FAIL = {"AC-004", "AC-005", "AC-006"}
GOLDEN_CONCERNS = {"AC-021", "AC-032", "AC-033"}

CATEGORIES = {
    "AC-003": ("out_of_scope_statements", "ambiguous_coverage", "The answer supplies the requested limit but adds a bodily-injury deductible claim outside the requested answer; the corpus wording does not clearly establish liability-specific deductible application.", "Limit answers should suppress unrelated deductible claims unless the question asks for them and the coverage-specific wording is explicit."),
    "AC-004": ("out_of_scope_statements", "incorrect_coverage_association", "The answer correctly gives the per-accident limit but adds a bodily-injury deductible that was not requested and associates the deductible with liability coverage.", "Constrain answers to the requested limit and test that deductible claims are not appended to liability-limit responses."),
    "AC-005": ("out_of_scope_statements", "incorrect_coverage_association", "The answer correctly gives the property-damage limit but adds a property-damage deductible that was not requested.", "Apply a scope check for limit questions and require explicit coverage-specific support before adding deductible statements."),
    "AC-006": ("out_of_scope_statements", "incorrect_coverage_association", "The answer correctly gives the property-damage limit but adds a deductible claim outside the requested answer.", "Apply the same scope and coverage-association guard used for other liability-limit questions."),
    "AC-023": ("unsupported_payments", "incorrect_abstention", "The answer presents property-damage limit and deductible values as relevant to a maximum collision payout, although the golden answer says no collision maximum is established.", "For payout questions, require a collision-specific limit before stating a maximum and explicitly abstain when it is absent."),
    "AC-027": ("policy_context_conflict", "unsupported_payments", "The answer makes a claim-denial statement using cited passages, then adds a qualification that approval cannot be determined from the schedule alone; this conflicts with the requested schedule-only abstention.", "Block claim decisions from schedule-only questions and require claim facts, coverage, exclusions, valuation, and documentation before payment or denial language."),
    "AC-029": ("ambiguous_coverage", "incorrect_coverage_association", "The answer lists general liability, collision, comprehensive, and deductible language but does not directly establish borrowed-vehicle coverage.", "Detect borrowed/non-owned vehicle questions and require an explicit non-owned or borrowed-vehicle provision before describing coverage."),
    "AC-031": ("unsupported_payments", "incorrect_abstention", "The answer calculates a €3,000 payment from repair cost minus deductible despite unresolved coverage applicability, valuation, exclusions, and claim facts.", "Require an abstention for payment calculations when the question supplies repair cost alone and no complete claim basis."),
    "AC-033": ("ambiguous_coverage", "incorrect_coverage_association", "The answer treats the schedule's €1,000 entries beside bodily-injury rows as an established liability deductible despite the conflicting scope of the contract deductible wording.", "Preserve the liability-deductible ambiguity and require human resolution before converting adjacent schedule values into coverage-specific conclusions."),
    "AC-034": ("incorrect_abstention", "out_of_scope_statements", "The answer lists general exclusions in response to a request for collision-specific exclusions, then says no collision-specific list is established.", "When only general exclusions are available, lead with the abstention and avoid presenting the general list as the requested collision-specific result."),
    "AC-035": ("policy_context_conflict", "incorrect_abstention", "The answer correctly says the passages do not contain the requested MOT-0000004 deductible but then supplies the MOT-0000002 deductible, mixing the selected policy and the policy named in the question.", "Enforce question-policy consistency and request clarification without providing a value from the conflicting policy."),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    prepared = {row["question_id"]: row for row in read_csv(PREPARED)}
    batch1 = {row["question_id"]: row for row in read_csv(BATCH_1)}
    batch2 = {row["question_id"]: row for row in read_csv(BATCH_2)}
    baseline = {row["question_id"]: row for row in read_csv(BASELINE)}
    traces = {row["question_id"]: row for row in json.loads(TRACES.read_text(encoding="utf-8"))}
    assert len(prepared) == len(baseline) == len(traces) == 40
    failure_ids = sorted(BATCH_1_FAIL | BATCH_2_FAIL, key=lambda value: int(value.split("-")[1]))
    assert len(failure_ids) == 11

    fields = [
        "question_id", "policy_id", "scenario", "expected_answer", "actual_answer", "evidence",
        "cited_filenames", "deterministic_indicators", "provisional_review_status",
        "independent_human_confirmation", "golden_interpretation_concern", "failure_description",
        "primary_category", "secondary_category", "recommended_improvement",
    ]
    rows = []
    for question_id in failure_ids:
        row = prepared[question_id]
        base = baseline[question_id]
        # Read the trace for provenance validation and to ensure this analysis is tied
        # to the frozen run; evidence text itself comes from the prepared worksheet.
        assert traces[question_id]["question_id"] == question_id
        source_review = batch1.get(question_id) or batch2.get(question_id)
        rows.append({
            "question_id": question_id,
            "policy_id": row["policy_id"],
            "scenario": row["scenario"],
            "expected_answer": row["expected_answer"],
            "actual_answer": row["actual_answer"],
            "evidence": row["evidence_excerpts"],
            "cited_filenames": row["cited_filenames"],
            "deterministic_indicators": row["deterministic_flags"],
            "provisional_review_status": "PROVISIONAL_FAIL",
            "independent_human_confirmation": "NO",
            "golden_interpretation_concern": "UNRESOLVED_GOLDEN_DATA_INTERPRETATION" if question_id in GOLDEN_CONCERNS else "",
            "failure_description": CATEGORIES[question_id][2],
            "primary_category": CATEGORIES[question_id][0],
            "secondary_category": CATEGORIES[question_id][1],
            "recommended_improvement": CATEGORIES[question_id][3],
        })
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    primary_counts = Counter(row["primary_category"] for row in rows)
    all_category_counts = Counter()
    for row in rows:
        all_category_counts[row["primary_category"]] += 1
        all_category_counts[row["secondary_category"]] += 1
    report = [
        "# Failure Analysis",
        "",
        "This report consolidates the frozen Week 4 baseline artifacts and provisional review signals. It does not create independently confirmed human labels, rerun the model, or modify RAG, golden data, Pinecone, baseline results, or traces.",
        "",
        "## Evidence and label boundaries",
        "",
        "- Deterministic baseline checks: 40/40 executed, 40/40 policy-isolated, and 84/84 citation references valid. These checks do not establish answer correctness or groundedness.",
        "- Provisional reviews: 20 total, reported as Batch 1 2 PASS / 8 FAIL and Batch 2 7 PASS / 3 FAIL. All remain provisional and `independent_human_confirmation=NO`.",
        "- Batch 1 has an explicit per-ID ledger. Batch 2 currently contains ten rows with `UNREVIEWED` judgment fields and no per-ID 7/3 allocation. The three Batch 2 failure IDs below were assigned from deterministic indicators and evidence comparison to make the supplied aggregate usable; they must not be treated as human-confirmed labels.",
        "- Unreviewed cases: 20, excluding the 20 provisional review IDs.",
        "- Ambiguous golden labels: AC-021, AC-032, and AC-033 remain explicitly unresolved.",
        "",
        "## Failure records",
        "",
        f"The taxonomy contains {len(rows)} distinct provisional failure cases. Each case is counted once by primary category; secondary categories describe additional risk and are not added to the primary-case total.",
        "",
        "| Category | Primary cases | Risk interpretation |",
        "|---|---:|---|",
    ]
    risk_text = {
        "out_of_scope_statements": "Extra claims increase review burden and can introduce unsupported policy conclusions.",
        "ambiguous_coverage": "Coverage-specific interpretation can create incorrect coverage or deductible advice.",
        "unsupported_payments": "Payment or denial language can cause direct claims-handling harm.",
        "incorrect_coverage_association": "A value from one coverage can be incorrectly transferred to another.",
        "policy_context_conflict": "Mixing policy IDs can produce an answer for the wrong insured policy.",
        "incorrect_abstention": "The system fails to abstain when the evidence is incomplete or conflicting.",
    }
    for category, count in primary_counts.most_common():
        report.append(f"| `{category}` | {count} | {risk_text[category]} |")
    report.extend([
        "",
        "## Top three patterns",
        "",
        "1. **Out-of-scope statements** — the most frequent pattern. Several limit or coverage answers add deductible, endorsement, or related-coverage claims not requested. This is a high insurance risk because extra claims can be read as coverage advice.",
        "2. **Incorrect coverage association / ambiguous coverage** — deductible or coverage language is attached to a coverage without a clear coverage-specific rule. This is especially material for AC-003, AC-032, and AC-033.",
        "3. **Unsupported payment or claim decisions** — AC-023, AC-027, and AC-031 include payout, denial, or settlement implications without sufficient claim-specific evidence. This is the highest direct claims-handling risk.",
        "",
        "## Recommended measurable targeted improvement",
        "",
        "Add a deterministic answer-scope and claims-decision guard for evaluation runs: for limit/deductible questions, flag any material coverage term or monetary value not requested; for payout/approval/denial questions, require an explicit evidence marker for the requested coverage and claim basis, otherwise require an abstention. Measure the next run with: (a) count of out-of-scope monetary/coverage claims among 40 cases, (b) count of unsupported payment/denial statements, (c) count of policy-context conflicts, and (d) verified abstention indicator on the abstention subset. Keep semantic correctness and citation support as separate human-review fields.",
        "",
        "The detailed evidence, expected and actual answers, categories, and recommended improvements are in [failure_taxonomy.csv](../results/failure_taxonomy.csv).",
    ])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"failure_cases": len(rows), "primary_counts": dict(primary_counts), "all_category_counts": dict(all_category_counts)})


if __name__ == "__main__":
    main()
