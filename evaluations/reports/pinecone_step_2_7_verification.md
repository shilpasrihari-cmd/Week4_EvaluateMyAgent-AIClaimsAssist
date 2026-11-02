# Week 4 Step 2.7 Pinecone Verification

This was a read-only verification after updating the Week 4 `.env`. No IDs, retrieval algorithms, indexes, vectors, golden data, or source documents were modified.

## Active index

The project-root environment resolves to:

- `PINECONE_INDEX_NAME`: `aiclaimsassist-vectors`
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- Pinecone key present: yes (Boolean check only)

The active index reports:

| Property | Observed value |
|---|---|
| Index | `aiclaimsassist-vectors` |
| Dimension | `384` |
| Metric | `cosine` |
| Ready | yes |
| `section-aware` vectors | `32` |
| `fixed-size` vectors | `10` |

## Semantic retrieval verification

All three queries returned five results, and every result matched the explicitly supplied policy ID:

| Case | Policy | Semantic results | Policy IDs returned | Top result |
|---|---|---:|---|---|
| AC-001 | MOT-0000002 | 5 | MOT-0000002 only | `MOT-0000002-schedule:section_aware:0002:57e50d4033` |
| AC-002 | MOT-0000004 | 5 | MOT-0000004 only | `MOT-0000004-schedule:section_aware:0002:e682c8c8a7` |
| AC-003 | MOT-0000002 | 5 | MOT-0000002 only | `MOT-0000002-declarations:section_aware:0002:9a10591741` |

The namespace, embedding dimension, and policy filters are therefore functioning against the populated index.

## Pinecone versus BM25 IDs

The semantic index uses IDs such as:

`MOT-0000002-schedule:section_aware:0002:57e50d4033`

The local BM25 corpus uses IDs such as:

`DOC-MOT-0000002-SCHEDULE:section_aware:0002:57e50d4033`

For all three cases, exact ID overlap was zero even when the final digest and section position identified the same underlying content. Examples include the AC-003 declarations, schedule, and contract chunks. The difference includes document-prefix and document-name normalization (`MOT-...` versus `DOC-MOT-...`, and `DEC` versus `declarations`).

## Impact

Current RRF deduplication keys on the complete `chunk_id`. Because equivalent semantic and BM25 chunks have different IDs, they are treated as separate candidates. The observed hybrid candidate lists contain paired semantic/keyword records for the same content, each receiving its own RRF contribution. This can:

- reduce the number of unique evidence passages represented in the top candidates;
- change RRF ordering and CrossEncoder input composition;
- consume candidate slots with duplicate content; and
- make exact-ID Recall@5 labels fail to match even when the source text is relevant.

Citation validation itself remains position-based: a generated `[Source N]` maps to the Nth reranked passage and then preserves that passage’s actual chunk ID and filename. Therefore ID-format mismatch does not by itself create a fabricated citation or invalidate a correctly bounded Source label. It does make citations from the two retrievers refer to separate IDs for equivalent content, which can complicate trace analysis and any exact chunk-ID-based evaluation.

## Recommended correction

The minimum configuration correction is already in place: use `aiclaimsassist-vectors`, the populated index, for semantic retrieval. Before relying on hybrid RRF metrics, the indexing and BM25 corpus must share one canonical chunk-ID contract. The preferred corrective action is to produce the Pinecone vectors and local BM25 chunks from the same current corpus and chunker, then verify that identical chunks have identical IDs in both stores. That is a future controlled indexing operation; it was not performed here.

Changing only the ID strings in evaluation output would hide the mismatch and is not recommended. No retrieval algorithm or index was altered during this verification.

## Test result

`python -m pytest -q` completed with **52 passed**. The all-40 evaluation was not run, and no credentials were exposed.
