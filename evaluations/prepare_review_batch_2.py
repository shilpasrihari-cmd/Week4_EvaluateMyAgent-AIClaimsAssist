"""Create the next human-review batch and preserve provisional judgments separately."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluations" / "results" / "human_review_prepared.csv"
BATCH = ROOT / "evaluations" / "results" / "review_batch_2.csv"
LEDGER = ROOT / "evaluations" / "results" / "provisional_review_ledger.csv"

PROVISIONAL_IDS = [
    "AC-003", "AC-021", "AC-023", "AC-027", "AC-029",
    "AC-031", "AC-032", "AC-033", "AC-034", "AC-035",
]
NEXT_BATCH_IDS = [
    "AC-001", "AC-002", "AC-004", "AC-005", "AC-006",
    "AC-007", "AC-008", "AC-009", "AC-010", "AC-011",
]
PROVISIONAL_PASS_IDS = {"AC-021", "AC-032"}
INTERPRETATION_CONCERN_IDS = {"AC-021", "AC-032", "AC-033"}


def load() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load()
    by_id = {row["question_id"]: row for row in rows}
    assert set(PROVISIONAL_IDS).issubset(by_id)
    assert set(NEXT_BATCH_IDS).issubset(by_id)

    batch_fields = [
        "question_id", "policy_id", "scenario", "question", "expected_answer", "actual_answer",
        "cited_filenames", "evidence_excerpts", "missing_information", "verification_status",
        "deterministic_flags", "golden_interpretation_concern", "answer_correctness",
        "citation_support", "groundedness", "failure_category", "reviewer_notes",
    ]
    with BATCH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_fields)
        writer.writeheader()
        for question_id in NEXT_BATCH_IDS:
            row = by_id[question_id]
            writer.writerow({
                field: row[field] if field in row and field not in {
                    "golden_interpretation_concern", "answer_correctness", "citation_support",
                    "groundedness", "failure_category", "reviewer_notes",
                } else ("UNRESOLVED_GOLDEN_DATA_INTERPRETATION" if field == "golden_interpretation_concern" and question_id in INTERPRETATION_CONCERN_IDS else "")
                for field in batch_fields
            } | {field: "UNREVIEWED" for field in batch_fields if field in {
                "answer_correctness", "citation_support", "groundedness", "failure_category", "reviewer_notes"
            }})

    ledger_fields = [
        "question_id", "policy_id", "scenario", "question", "expected_answer", "actual_answer",
        "cited_filenames", "evidence_excerpts", "deterministic_flags", "verification_status",
        "provisional_judgment", "provisional_judgment_status", "golden_interpretation_concern",
        "independent_human_confirmation",
    ]
    with LEDGER.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        writer.writeheader()
        for question_id in PROVISIONAL_IDS:
            row = by_id[question_id]
            writer.writerow({
                "question_id": question_id,
                "policy_id": row["policy_id"],
                "scenario": row["scenario"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "actual_answer": row["actual_answer"],
                "cited_filenames": row["cited_filenames"],
                "evidence_excerpts": row["evidence_excerpts"],
                "deterministic_flags": row["deterministic_flags"],
                "verification_status": row["verification_status"],
                "provisional_judgment": "PASS" if question_id in PROVISIONAL_PASS_IDS else "FAIL",
                "provisional_judgment_status": "PROVISIONAL_ONLY",
                "golden_interpretation_concern": "UNRESOLVED_GOLDEN_DATA_INTERPRETATION" if question_id in INTERPRETATION_CONCERN_IDS else "",
                "independent_human_confirmation": "NO",
            })

    print(f"batch_rows={len(NEXT_BATCH_IDS)} provisional_rows={len(PROVISIONAL_IDS)}")
    print("batch_ids=" + ",".join(NEXT_BATCH_IDS))
    print("provisional_ids=" + ",".join(PROVISIONAL_IDS))
    print("provisional_pass=2 provisional_fail=8")


if __name__ == "__main__":
    main()
