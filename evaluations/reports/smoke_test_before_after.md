# Week 4 Step 2.5 Smoke Test Before/After

This comparison covers only AC-001, AC-002, and AC-003. The pre-fix artifacts were preserved as [diagnostic_traces_before_step_2_5.json](../results/diagnostic_traces_before_step_2_5.json), [baseline_results_before_step_2_5.csv](../results/baseline_results_before_step_2_5.csv), and [baseline_traces_before_step_2_5.json](../results/baseline_traces_before_step_2_5.json). No golden data, source document, Pinecone index, prompt, or full 40-case evaluation was changed or run.

## Comparison

| ID | Before | After | Citation validity | Policy isolation | Latency seconds | Errors |
|---|---|---|---|---|---:|---|
| AC-001 | Answer was suppressed and had 0 valid citations. | `The deductible is €1,000.00 [Source 1].` | 1 valid citation; schedule chunk mapping preserved. | `true` | 3.215991 | none |
| AC-002 | Answer was suppressed and had 0 valid citations. | `The deductible is €500.00 [Source 1].` | 1 valid citation; schedule chunk mapping preserved. | `true` | 1.971070 | none |
| AC-003 | Answer retained with 1 valid citation; the combined citation was only partially recognized. | `The bodily injury liability limit per person is €100,000.00 [Source 1]. It also appears with a €1,000.00 deductible [Source 2].` | 2 valid citations; declarations and schedule chunk mappings preserved. | `true` | 2.115035 | none |

The displayed currency characters follow the project’s existing document decoding in the CSV/trace output. The source mapping and citation IDs are unchanged and are the ground for the validity checks.

## Targeted fixes

Generic deductible questions now retain a supported generic deductible statement while removing category-specific liability wording from the final answer. Coverage-specific deductible questions continue to remove sentences for unrelated coverages. This prevents the previous all-sentences removal that caused AC-001 and AC-002 to be suppressed, without asserting a liability-specific deductible in the generic answer.

Citation validation now accepts both `[Source N]` and combined forms such as `[Source 1, Source 2]`. Each recognized label is mapped independently to the ordered reranked passage list. Out-of-range or fabricated labels remain invalid and cannot create citations.

## Semantic retrieval diagnosis

The diagnostic probe used the same requested settings as the live pipeline:

- namespace: `section-aware`
- policy filter: `{"policy_id": {"$eq": "MOT-0000002"}}` or `{"policy_id": {"$eq": "MOT-0000004"}}`
- expected and observed query embedding dimension: `384`
- observed normalized query-vector norm: approximately `1.0`
- semantic query match count: `0` for all three cases
- local policy-ID validation: passed for all three policies
- Pinecone index statistics: `fixed-size` had `0` vectors; `section-aware` was absent from the returned namespace map

The Pinecone data-plane `Index` object does not expose `describe_index`, so index dimension and metric could not be confirmed through that object. The probe surfaced this as a sanitized diagnostic note rather than guessing. The observed namespace statistics are consistent with the empty semantic results. No reindexing or index mutation was performed.

## Validation

The project test suite passed with **52 tests**. The live smoke run completed all three cases with no recorded API or deterministic integrity errors. Semantic answer correctness remains for human review; this report evaluates answer availability, citation validity, source mapping, policy isolation, latency, and surfaced errors only.
