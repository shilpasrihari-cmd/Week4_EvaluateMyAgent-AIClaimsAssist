# Human Review Guide

This worksheet was prepared from the frozen golden dataset, baseline results, baseline traces, and existing human-review template. No model judge, semantic string comparison, or human judgment was run.

- Worksheet rows: 40
- VERIFIED: 37
- NEEDS_REVIEW: 3
- Deterministically flagged rows: 30
- All human judgment fields are `UNREVIEWED`.

## Review priority

Priority 1 is the three golden cases already marked NEEDS_REVIEW: AC-021, AC-032, and AC-033. Their expected answers were not reclassified here. Review whether the source evidence justifies the expected abstention or answer before using them for evaluation conclusions.

Priority 2 is every row with a value other than `NONE_OBSERVED` in `deterministic_flags`. These are screening indicators for human inspection, not correctness judgments.

### Deterministic screening summary
- `missing_abstention`: 15 row(s): AC-007, AC-008, AC-009, AC-020, AC-021, AC-023, AC-024, AC-027, AC-029, AC-031, AC-032, AC-033, AC-035, AC-037, AC-040
- `potential_conflicting_sources`: 4 row(s): AC-025, AC-027, AC-028, AC-039
- `potential_out_of_scope_statement`: 18 row(s): AC-003, AC-004, AC-005, AC-006, AC-007, AC-010, AC-012, AC-015, AC-017, AC-019, AC-022, AC-023, AC-024, AC-027, AC-029, AC-030, AC-034, AC-039
- `potential_unsupported_coverage`: 12 row(s): AC-003, AC-004, AC-005, AC-006, AC-015, AC-022, AC-027, AC-030, AC-032, AC-034, AC-037, AC-040

## Reviewer instructions

1. Compare the actual answer with the expected answer and the cited source excerpt; record `PASS`, `FAIL`, or `UNCERTAIN` in `answer_correctness`.
2. Record whether each cited filename, section, and excerpt supports the material claim in `citation_support`.
3. Record whether the answer is grounded in the cited evidence, including any extra claim, unsupported coverage statement, or missing abstention, in `groundedness`.
4. Choose a failure category only after inspecting the evidence. Use reviewer notes for the exact source text or reasoning.
5. Treat `unsupported_citations`, `missing_citations`, `missing_abstention`, `potential_out_of_scope_statement`, `potential_unsupported_coverage`, and `potential_conflicting_sources` as prompts for inspection. They do not establish that the answer is wrong.

The worksheet preserves question IDs, policy IDs, verification statuses, expected answers, generated answers, citation filenames, and evidence excerpts. The original baseline artifacts and RAG pipeline were not changed.
