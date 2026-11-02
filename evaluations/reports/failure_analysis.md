# Failure Analysis

This report consolidates the frozen Week 4 baseline artifacts and provisional review signals. It does not create independently confirmed human labels, rerun the model, or modify RAG, golden data, Pinecone, baseline results, or traces.

## Evidence and label boundaries

- Deterministic baseline checks: 40/40 executed, 40/40 policy-isolated, and 84/84 citation references valid. These checks do not establish answer correctness or groundedness.
- Provisional reviews: 20 total, reported as Batch 1 2 PASS / 8 FAIL and Batch 2 7 PASS / 3 FAIL. All remain provisional and `independent_human_confirmation=NO`.
- Batch 1 has an explicit per-ID ledger. Batch 2 currently contains ten rows with `UNREVIEWED` judgment fields and no per-ID 7/3 allocation. The three Batch 2 failure IDs below were assigned from deterministic indicators and evidence comparison to make the supplied aggregate usable; they must not be treated as human-confirmed labels.
- Unreviewed cases: 20, excluding the 20 provisional review IDs.
- Ambiguous golden labels: AC-021, AC-032, and AC-033 remain explicitly unresolved.

## Failure records

The taxonomy contains 11 distinct provisional failure cases. Each case is counted once by primary category; secondary categories describe additional risk and are not added to the primary-case total.

| Category | Primary cases | Risk interpretation |
|---|---:|---|
| `out_of_scope_statements` | 4 | Extra claims increase review burden and can introduce unsupported policy conclusions. |
| `unsupported_payments` | 2 | Payment or denial language can cause direct claims-handling harm. |
| `policy_context_conflict` | 2 | Mixing policy IDs can produce an answer for the wrong insured policy. |
| `ambiguous_coverage` | 2 | Coverage-specific interpretation can create incorrect coverage or deductible advice. |
| `incorrect_abstention` | 1 | The system fails to abstain when the evidence is incomplete or conflicting. |

## Top three patterns

1. **Out-of-scope statements** — the most frequent pattern. Several limit or coverage answers add deductible, endorsement, or related-coverage claims not requested. This is a high insurance risk because extra claims can be read as coverage advice.
2. **Incorrect coverage association / ambiguous coverage** — deductible or coverage language is attached to a coverage without a clear coverage-specific rule. This is especially material for AC-003, AC-032, and AC-033.
3. **Unsupported payment or claim decisions** — AC-023, AC-027, and AC-031 include payout, denial, or settlement implications without sufficient claim-specific evidence. This is the highest direct claims-handling risk.

## Recommended measurable targeted improvement

Add a deterministic answer-scope and claims-decision guard for evaluation runs: for limit/deductible questions, flag any material coverage term or monetary value not requested; for payout/approval/denial questions, require an explicit evidence marker for the requested coverage and claim basis, otherwise require an abstention. Measure the next run with: (a) count of out-of-scope monetary/coverage claims among 40 cases, (b) count of unsupported payment/denial statements, (c) count of policy-context conflicts, and (d) verified abstention indicator on the abstention subset. Keep semantic correctness and citation support as separate human-review fields.

The detailed evidence, expected and actual answers, categories, and recommended improvements are in [failure_taxonomy.csv](../results/failure_taxonomy.csv).
