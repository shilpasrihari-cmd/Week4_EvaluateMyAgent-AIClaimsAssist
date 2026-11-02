# Week 4 smoke-test diagnosis

This diagnosis covers only AC-001, AC-002, and AC-003 from the three-question
smoke test. No 40-question evaluation was run. The report uses the saved
baseline traces plus a focused diagnostic run of the existing semantic,
BM25/hybrid, and CrossEncoder components. No credentials are included.

## Summary

The failures for AC-001 and AC-002 did not originate from policy filtering or
from the final reranking stage. Pinecone semantic retrieval returned zero
matches for both questions, so the hybrid retriever fell back to BM25-only
results. BM25 did retrieve the correct policy schedule, and the CrossEncoder
promoted it to rank 1 of the final five passages. The pipeline then returned a
generated fallback because no citations survived validation. The raw model
response and rejected citation labels were not recorded, so the precise model
citation behavior is **UNCONFIRMED**.

AC-003 retrieved the correct declarations and schedule passages. The answer
included the requested EUR 100,000 bodily-injury limit, but also added a EUR
1,000 bodily-injury deductible statement. The schedule visibly places EUR
1,000 beside the bodily-injury row, while the contract wording explicitly ties
the applicable deductible to collision and comprehensive coverage. Therefore
the table supports that the schedule displays the value, but whether it should
be interpreted as a liability-specific deductible is ambiguous and requires
human review.

## Findings

| ID | Expected answer | Retrieval status | Generation status | Citation validation status | Root cause | Evidence | Recommended fix |
|---|---|---|---|---|---|---|---|
| AC-001 | EUR 1,000 per occurrence | **Retrieved and reranked.** Semantic: 0 matches. BM25 returned the policy schedule at rank 4 with score 7.885407. Hybrid RRF retained it at rank 4 with score 0.015625. CrossEncoder moved it to rank 1 with score -1.558596. It was in the final five. | The saved result is the pipeline fallback: `I cannot provide a verifiable grounded answer from the retrieved evidence.` No API failure occurred. Whether the LLM initially answered is **UNCONFIRMED**. | No validated citations. `missing_citations=true`; `citation_ids_ok=true` is vacuous because there were no citation objects to validate. No invalid citation labels are preserved. | Citation failure after successful evidence retrieval. The exact failure point inside model output versus citation parsing cannot be determined from the trace. The pipeline suppresses the answer whenever validated citations are empty or invalid. | Final trace contains the schedule chunk `DOC-MOT-0000002-SCHEDULE:section_aware:0002:57e50d4033` as final rank 1. Its text contains `Deductible ... EUR 1,000.00`. The trace has `citations: []`, generic missing information, and no `error`. | Record the raw LLM response, prompt, parsed citation labels, invalid labels, and reranker scores in traces. Then inspect whether the model omitted citations or used a syntax the validator rejected. |
| AC-002 | EUR 500 per occurrence | **Retrieved and reranked.** Semantic: 0 matches. BM25 returned the policy schedule at rank 4 with score 8.028306. Hybrid RRF retained it at rank 4 with score 0.015625. CrossEncoder moved it to rank 1 with score -1.728844. It was in the final five. | The saved result is the same pipeline fallback. No API failure occurred. Whether the LLM initially answered is **UNCONFIRMED**. | No validated citations. `missing_citations=true`; `citation_ids_ok=true` is vacuous. No rejected citation labels are preserved. | Same post-generation citation failure as AC-001. Retrieval and reranking supplied the correct schedule, so the trace does not support a retrieval-loss explanation. | Final trace contains `DOC-MOT-0000004-SCHEDULE:section_aware:0002:e682c8c8a7` at final rank 1. Its text contains `Deductible ... EUR 500.00`. The trace has `citations: []`, generic missing information, and no `error`. | Same observability fix: preserve raw response and citation-parser details before the pipeline fallback replaces the answer. |
| AC-003 | EUR 100,000 bodily-injury limit per person | **Retrieved and reranked.** Semantic: 0 matches. BM25 ranked declarations first with score 19.065189 and schedule second with score 15.563330. Hybrid RRF kept that order at 0.016393 and 0.016129. CrossEncoder kept declarations rank 1 with score 7.022536 and schedule rank 2 with score 6.227262. Both were in the final five. | The answer was generated and returned without API error. It answered the requested limit, then added a deductible statement for bodily injury. | One citation validated: Source 2, the schedule chunk. The answer text used `[Source 1, Source 2]` for the limit and `[Source 2]` for the deductible. The validator recognizes exact labels such as `[Source 1]`; it did not validate Source 1 from the combined comma form. The answer was not suppressed because Source 2 validated. | Two issues: the model over-answered with a related deductible, and its combined citation syntax was only partially recognized. The schedule displays EUR 1,000 beside bodily injury, but the contract says collision and comprehensive are subject to the applicable deductible; liability-specific application is ambiguous. | Validated Source 2 is `DOC-MOT-0000002-SCHEDULE:section_aware:0002:57e50d4033`, whose text displays `Bodily injury – per person / EUR 100,000.00 / EUR 1,000.00`. Source 1 is the declarations chunk `DOC-MOT-0000002-DEC:section_aware:0002:9a10591741`, which states the EUR 100,000 limit but no deductible. The contract chunk `DOC-MOT-0000002-CONTRACT:section_aware:0002:717fb75477` states that collision and comprehensive coverages apply less the applicable deductible. | Human-review the liability-deductible interpretation. For observability, record raw citation labels and validation rejects. Semantic answer correctness must remain a judge or human-review decision. |

## Retrieval and ranking detail

The baseline trace records only the final five passages. The focused diagnostic
run recovered the intermediate values for these three questions:

- **AC-001:** BM25 order was contract header, exclusions, C-1003 denial,
  schedule, declarations. Semantic returned no matches. The CrossEncoder
  reordered the schedule to rank 1, followed by exclusions, contract header,
  declarations, and the denial letter.
- **AC-002:** BM25 order was contract header, C-1002 denial, exclusions,
  schedule, and C-1004 FNOL. Semantic returned no matches. The CrossEncoder
  reordered the schedule to rank 1, followed by contract header, exclusions,
  FNOL, and denial letter.
- **AC-003:** BM25 order was declarations, schedule, contract insuring
  agreement, declarations endorsements, and C-1003 adjuster report. Semantic
  returned no matches. The CrossEncoder retained declarations, schedule, and
  contract as the first three.

The semantic result lists were empty, but the existing `HybridRetriever.search`
implementation catches semantic-search exceptions silently in hybrid mode. The
observed empty result therefore proves that no semantic matches reached fusion,
but does not prove whether Pinecone returned an empty match list or a semantic
query exception was swallowed. The trace has no semantic exception field.

## Missing observability fields

The current baseline traces are insufficient to distinguish model citation
omission from citation-parser rejection for AC-001 and AC-002. They do not
contain:

- the raw Nebius response before citation validation and fallback suppression;
- the constructed prompt and ordered `[Source N]` passage map;
- the semantic candidate list or Pinecone response metadata;
- BM25 candidates and scores;
- RRF candidates and scores;
- CrossEncoder scores before the final five are returned;
- invalid citation labels returned by `RAGPipeline.validate_citations`;
- a field stating whether the final answer was original model text or a
  pipeline-generated fallback.

The saved trace does establish the final retrieved evidence, policy IDs,
validated citations, missing-information messages, model name, latency, and
deterministic policy/citation checks. No semantic correctness judgment is made
in this report.
