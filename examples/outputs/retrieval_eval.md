# Retrieval evaluation

Labelled questions from `examples/questions.json`. chunk_size=1000, overlap=200, embed=BAAI/bge-small-en-v1.5, candidate_k=20, rrf_k=60.

## Page-level (15 questions)

Hit if any retrieved chunk comes from a page that contains the answer.

| mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| dense | 0.60 | 0.87 | 0.93 | 0.724 |
| bm25 | 0.53 | 0.93 | 1.00 | 0.717 |
| hybrid | 0.53 | 1.00 | 1.00 | 0.744 |

## Chunk-level (14 questions)

Hit only if a retrieved chunk *itself* contains the answer strings (`evidence_must_contain`). This is what the LLM actually needs; page-level can count a neighbouring chunk on the same page as a hit.

| mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| dense | 0.57 | 0.86 | 0.86 | 0.714 |
| bm25 | 0.50 | 0.71 | 0.79 | 0.613 |
| hybrid | 0.43 | 0.79 | 0.86 | 0.621 |

## Per question: rank of first evidence chunk (top-5, `miss` = not retrieved)

| id | type | dense | bm25 | hybrid |
|---|---|---|---|---|
| q01 | factual | miss | miss | miss |
| q02 | factual | 2 | 1 | 2 |
| q03 | factual | 2 | 4 | 2 |
| q04 | factual | 1 | miss | 5 |
| q05 | factual | 1 | 1 | 2 |
| q06 | factual | 1 | 3 | 2 |
| q07 | conceptual | 1 | 2 | 1 |
| q08 | conceptual | 1 | 2 | 1 |
| q09 | paraphrase | 1 | 1 | 1 |
| q10 | conceptual | 1 | 1 | 1 |
| q11 | factual | 2 | 1 | 1 |
| q12 | factual | 2 | 1 | 2 |
| q13 | conceptual | 1 | 1 | 1 |
| q14 | factual | miss | miss | miss |

## Per question: rank of first chunk from a relevant page

| id | type | dense | bm25 | hybrid |
|---|---|---|---|---|
| q01 | factual | 1 | 3 | 1 |
| q02 | factual | 2 | 1 | 2 |
| q03 | factual | 1 | 3 | 1 |
| q04 | factual | 1 | 1 | 1 |
| q05 | factual | 1 | 1 | 1 |
| q06 | factual | 1 | 3 | 2 |
| q07 | conceptual | 1 | 1 | 1 |
| q08 | conceptual | 5 | 2 | 2 |
| q09 | paraphrase | 1 | 1 | 1 |
| q10 | conceptual | 1 | 2 | 1 |
| q11 | factual | 3 | 1 | 2 |
| q12 | factual | 2 | 1 | 2 |
| q13 | conceptual | 1 | 1 | 1 |
| q14 | factual | 3 | 2 | 3 |
| q15 | cross-document | miss | 4 | 3 |
