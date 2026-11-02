# Week 4 Baseline Evaluation Report

The run evaluated 40 questions sequentially: 37 VERIFIED and 3 NEEDS_REVIEW.
Semantic answer correctness, citation support, and groundedness are left UNREVIEWED in the human-review queue.

## Frozen configuration

- Git commit: `3646874c2db65ec67e31c798f746c7e43a05608e`; working tree dirty at run time: `True`.
- Pinecone index: `aiclaimsassist-vectors`.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Namespaces: `section-aware` and `fixed-size`.
- Retrieval: hybrid, section-aware, RRF k=60, up to 10 candidates, top 5 passages, CrossEncoder reranking.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- LLM: `google/gemma-3-27b-it`.

## Deterministic metrics

- Execution success: 40/40 (1.0000).
- Policy isolation: 40/40 (1.0000).
- Citation reference validity: 84/84 (1.0).
- Citation coverage: 40/40 (1.0).
- Verified abstention indicator: 2/14 (0.14285714285714285).
- Recall@5: N/A (0/0); the golden dataset has no verified relevant chunk-ID labels.
- Latency seconds: mean 1.272573, median 1.246853, P95 1.579888, denominator 40.
- Error rate: 0/40 (0.0000).
- Missing-citation cases: 0/40; IDs: none.

## Review queue

NEEDS_REVIEW cases: AC-021, AC-032, AC-033.
All 40 rows in `human_review.csv` have UNREVIEWED answer correctness, citation support, groundedness, failure category, and notes fields.

## Errors

No errors were recorded.
