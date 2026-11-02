# Answer Scope Improvement: AC-003 through AC-006

This comparison covers only AC-003, AC-004, AC-005, and AC-006. The original values come from the frozen [baseline_results.csv](../results/baseline_results.csv). The improved values come from the separate live smoke-test run in [scope_smoke_results.csv](../results/scope_smoke_results.csv), with detailed traces in [scope_smoke_traces.json](../results/scope_smoke_traces.json). The 40-case baseline, golden dataset, Pinecone index, retrieval algorithms, and model configuration were not changed.

## Change

The generation pipeline now applies a question-driven scope guard after generation: when the question does not request deductible information, deductible sentences are removed before citation validation. Deductible questions retain their existing generic and coverage-specific handling. The prompt, citation parser, source mapping, policy filtering, and retrieval algorithms were not changed.

## Before and after

| ID | Frozen original answer | Improved answer | Scope result | Citations | Policy isolation | Latency seconds | Errors |
|---|---|---|---|---|---|---:|---|
| AC-003 | The bodily injury liability limit per person is €100,000.00 [Source 1, Source 2]. There is a €1,000.00 deductible for bodily injury per person [Source 2]. | The bodily injury liability limit per person is €100,000.00 [Source 1, Source 2]. | No unsolicited deductible | 2 valid; 0 invalid | PASS | 4.383783 | None |
| AC-004 | The bodily injury liability limit per accident is €300,000.00 [Source 1]. The deductible for bodily injury per accident is €500.00 [Source 2]. | The bodily injury liability limit per accident is €300,000.00 [Source 1]. | No unsolicited deductible | 1 valid; 0 invalid | PASS | 2.941765 | None |
| AC-005 | The property damage liability limit is €100,000.00 [Source 1]. The property damage deductible is €1,000.00 [Source 2]. | The property damage liability limit is €100,000.00 [Source 1]. | No unsolicited deductible | 1 valid; 0 invalid | PASS | 7.157851 | None |
| AC-006 | The property damage liability limit is €100,000.00 [Source 1]. The deductible for property damage is €500.00 [Source 2]. | The property damage liability limit is €100,000.00 [Source 1]. | No unsolicited deductible | 1 valid; 0 invalid | PASS | 4.384482 | None |

## Measured comparison

- Scope compliance: 0/4 original answers were free of the flagged unsolicited deductible statements; 4/4 improved answers were free of them.
- Execution: 4/4 improved cases returned answers.
- Citation validity: 4/4 improved cases passed deterministic citation-ID validation; 5 total citation references were accepted and 0 rejected.
- Policy isolation: 4/4 improved cases passed policy isolation.
- Errors: 0/4 improved cases recorded errors or API failures.
- Latency: original mean 1.167957 seconds; improved mean 4.716970 seconds. The smoke runs were separate live calls, so this is an observed comparison rather than a controlled latency benchmark.

The AC-003 trace confirms both `[Source 1]` and `[Source 2]` were accepted after the scope filter. All four traces record the deductible claim as removed, with fallback status `none`.

## Remaining issues and limits

The improvement addresses the observed out-of-scope deductible pattern for these four limit questions. It does not make a semantic correctness or source-support judgment; those remain human-review decisions. It also does not resolve the separate ambiguous liability-deductible interpretation concerns in AC-032 and AC-033. No all-40 evaluation was run.

## Tests

- Focused RAG tests: 15 passed.
- Full test suite: 57 passed.
- Added regression coverage for liability-limit scope, property-limit scope, citation preservation, generic deductibles, policy isolation, and existing ambiguous/missing-evidence behavior.
