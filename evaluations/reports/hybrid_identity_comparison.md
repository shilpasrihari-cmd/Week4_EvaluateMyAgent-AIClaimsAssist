# Week 4 Step 2.8 Hybrid Identity Comparison

This change was validated against the active `aiclaimsassist-vectors` index and only AC-001, AC-002, and AC-003. The pre-change artifacts were preserved as [diagnostic_traces_before_step_2_8.json](../results/diagnostic_traces_before_step_2_8.json), [baseline_results_before_step_2_8.csv](../results/baseline_results_before_step_2_8.csv), and [baseline_traces_before_step_2_8.json](../results/baseline_traces_before_step_2_8.json). No vectors, source documents, golden data, or Pinecone indexes were modified.

## Identity rule

Hybrid fusion now computes a stable canonical identity from all of these fields:

1. uppercased policy ID;
2. normalized filename;
3. normalized section heading; and
4. normalized full chunk text.

Normalization applies Unicode normalization, case folding, and whitespace collapsing. It does not strip `DOC-`, rename document types, ignore filenames, or ignore text boundaries. A merge occurs only when the policy, filename, section, and normalized full text all match. Different text, boundaries, policy IDs, filenames, or sections produce different canonical identities.

Each merged result preserves:

- the original selected `chunk_id`;
- every contributing original ID in `source_chunk_ids`;
- the canonical identity in `canonical_id`; and
- the original semantic/BM25 scores used to calculate the combined RRF score.

## Verified equivalence

For the live candidates, equivalent pairs had matching policy IDs, filenames, sections, normalized text, and digest content. Examples were the schedule coverage chunks for AC-001 and AC-002 and the declarations, schedule, and contract coverage chunks for AC-003. The exact full IDs differed because one store used `MOT-...`/`declarations` identifiers and the other used `DOC-MOT-...`/`DEC` identifiers. The identity implementation merged these only after metadata and text comparison verified equivalence.

## Same-index ranking comparison

The legacy calculation below uses complete original chunk IDs as keys. The canonical calculation uses the new verified identity. Both use the same semantic and BM25 candidates from the active index.

| Case | Legacy candidate count | Canonical candidate count | Verified merges | Canonical top RRF result | Top reranked result | Valid citations | Latency seconds | Errors | Policy isolation |
|---|---:|---:|---:|---|---|---:|---:|---|---|
| AC-001 | 10 | 8 | 2 | Schedule coverage, `0.032018` | Schedule deductible chunk, rank 1 | 3 | 7.160446 | none | true |
| AC-002 | 10 | 9 | 1 | Schedule coverage, `0.032018` | Schedule deductible chunk, rank 1 | 3 | 6.129172 | none | true |
| AC-003 | 10 | 7 | 3 | Declarations `0.032787`; schedule `0.032258`; contract `0.031746` | Declarations coverage, rank 1 | 2 | 2.382709 | none | true |

The canonical RRF score for an equivalent pair is the sum of its semantic and BM25 contributions. For example, a rank-1 semantic result merged with a rank-4 keyword result receives the two contributions associated with those ranks rather than occupying two candidate slots.

## Smoke results

- AC-001: `The deductible for this policy is €1,000.00 [Source 1], [Source 3]. [Source 2]`. Three citations validated; the schedule coverage citation retains both the Pinecone and BM25 original IDs.
- AC-002: `The deductible (per occurrence) is €500.00 [Source 1]. [Source 2] [Source 3]`. Three citations validated; the schedule coverage citation retains both original IDs.
- AC-003: `The bodily injury liability limit per person is €100,000.00 [Source 1, Source 2]. There is a €1,000.00 deductible for bodily injury per person [Source 2].` Two citations validated; declarations and schedule mappings retain their contributing original IDs.

The live run reported no deterministic errors, empty responses, API failures, or policy-isolation violations. Semantic answer correctness remains a human-review judgment.

## Citation and trace impact

Citation labels continue to map by ordered reranked Source number. The citation payload now includes `canonical_id` and `source_chunk_ids`, so a reviewer can see both the identity used for RRF and the original Pinecone/BM25 IDs that contributed to the evidence. Fabricated or out-of-range labels remain rejected by the existing bounds checks. The original IDs and retrieval scores are not rewritten.

## Tests and scope

The test suite completed with **55 passed**. Added coverage includes equivalent-ID merges, different text/boundaries, policy isolation, canonical/RRF score behavior, original-ID preservation, and citation mapping. Only the three smoke questions were run; the all-40 evaluation was not run.
