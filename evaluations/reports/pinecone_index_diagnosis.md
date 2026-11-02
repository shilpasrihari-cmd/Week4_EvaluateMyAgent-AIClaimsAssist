# Week 4 Step 2.6 Pinecone Vector Index Diagnosis

This was a read-only audit. No index, namespace, vector, source document, RAG code, or golden dataset was changed.

## Configuration comparison

| Check | Week 4 | Week 2 | Result |
|---|---|---|---|
| `PINECONE_INDEX_NAME` | `aiclaimsassist` | `aiclaimsassist` | Same selected index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | Same model |
| `PINECONE_API_KEY` present | yes | yes | Boolean check only |
| `NEBIUS_API_KEY` present | yes | yes | Boolean check only |
| Pinecone credential equality | yes | — | Both files resolve to the same Pinecone account/configuration |

The Week 4 `.env` therefore points to the same index name as the original Week 2 `.env`; it does not point to a separate Week 4 index.

## Pinecone inventory

Both indexes were ready and reported the expected vector shape/configuration:

| Index | Dimension | Metric | Namespace statistics |
|---|---:|---|---|
| `aiclaimsassist` | 384 | cosine | `fixed-size: 0`; `section-aware` absent |
| `aiclaimsassist-vectors` | 384 | cosine | `section-aware: 32`; `fixed-size: 10` |

The active Week 4 index is `aiclaimsassist`. Its empty namespace state explains why the semantic retriever returned zero matches and why hybrid retrieval fell back to BM25-only behavior. The query embedding itself was compatible: the live diagnostic trace recorded dimension `384` and a normalized vector norm of approximately `1.0`.

## Indexing and namespace checks

The project indexing script calls `create_indexer()` and `index_all()`. The indexer uses `sentence-transformers/all-MiniLM-L6-v2`, validates dimension `384`, and writes both namespaces:

- `fixed_size` → `fixed-size`
- `section_aware` → `section-aware`

The retrieval code queries `section-aware` for the requested `section_aware` strategy and applies an exact `policy_id` metadata filter. The trace showed the expected filters for MOT-0000002 and MOT-0000004, and the local policy-ID checks passed.

As a control, the same read-only semantic queries against `aiclaimsassist-vectors` returned five matches for each smoke question. Every returned result had the requested policy ID:

- AC-001 / MOT-0000002: 5 matches, top score approximately 0.561948.
- AC-002 / MOT-0000004: 5 matches, top score approximately 0.570733.
- AC-003 / MOT-0000002: 5 matches, top score approximately 0.648938.

This confirms that the embedding model, namespace spelling, and policy filter are capable of returning semantic results when pointed at the populated index. The populated index uses chunk IDs with an `MOT-...` prefix, while the current local BM25 traces use `DOC-MOT-...` IDs; its source metadata and policy IDs were still present and policy-filtered. That prefix difference should be checked before adopting the populated index as the canonical source for mixed retrieval.

## Diagnosis

The evidence supports **wrong or empty index selection**, rather than an embedding-dimension or metric incompatibility:

1. Week 4 and Week 2 configuration select `aiclaimsassist`.
2. `aiclaimsassist` is ready, 384-dimensional, cosine, but has no `section-aware` vectors and zero `fixed-size` vectors.
3. `aiclaimsassist-vectors` is ready with the same dimension and metric and contains both expected namespaces.
4. Identical model-generated 384-dimensional normalized queries return policy-isolated results from `aiclaimsassist-vectors`.

The available evidence does not prove whether `aiclaimsassist` was intentionally cleared, created without uploads, or was meant to be a different environment. It does establish that the vectors are present in the separate `aiclaimsassist-vectors` index and missing from the configured `aiclaimsassist` index.

## Minimum corrective action

Confirm with the index owner that `aiclaimsassist-vectors` is the intended standard vector index. If confirmed, change only the local `PINECONE_INDEX_NAME` value in the Week 4 environment from `aiclaimsassist` to `aiclaimsassist-vectors`, restart the application/evaluation process, and run a small semantic smoke check. No reindexing is needed for that index based on the observed namespace counts.

If `aiclaimsassist` is the required canonical name, the minimum action is an authorized vector population/rebuild into its two namespaces using the existing indexer and matching corpus; that is a separate mutating operation and was not performed here.

No credentials were printed. The all-40 evaluation was not run.
