# Chunk-size sweep

Overlap = 20% of chunk size; all other settings default. Same labelled questions as `retrieval_eval.md`. Caveats: larger chunks cover more text, so a top-5 of 1500-char chunks shows the LLM ~3x more text than a top-5 of 500-char chunks (and is more likely to hit the context cap); with only 15 questions, one question moves Hit@k by 0.067.

| chunk size | chunks |
|---|---|
| 500 | 540 |
| 1000 | 301 |
| 1500 | 224 |

## Page-level

| chunk size / mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| 500 / dense | 0.67 | 0.80 | 0.87 | 0.739 |
| 500 / bm25 | 0.60 | 0.80 | 0.93 | 0.727 |
| 500 / hybrid | 0.80 | 0.87 | 1.00 | 0.863 |
| 1000 / dense | 0.60 | 0.87 | 0.93 | 0.724 |
| 1000 / bm25 | 0.53 | 0.93 | 1.00 | 0.717 |
| 1000 / hybrid | 0.53 | 1.00 | 1.00 | 0.744 |
| 1500 / dense | 0.73 | 0.87 | 0.93 | 0.817 |
| 1500 / bm25 | 0.53 | 0.87 | 1.00 | 0.711 |
| 1500 / hybrid | 0.73 | 1.00 | 1.00 | 0.867 |

## Chunk-level (retrieved chunk contains the answer strings)

| chunk size / mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| 500 / dense | 0.29 | 0.50 | 0.57 | 0.387 |
| 500 / bm25 | 0.36 | 0.50 | 0.79 | 0.477 |
| 500 / hybrid | 0.43 | 0.64 | 0.79 | 0.556 |
| 1000 / dense | 0.57 | 0.86 | 0.86 | 0.714 |
| 1000 / bm25 | 0.50 | 0.71 | 0.79 | 0.613 |
| 1000 / hybrid | 0.43 | 0.79 | 0.86 | 0.621 |
| 1500 / dense | 0.64 | 0.86 | 0.93 | 0.740 |
| 1500 / bm25 | 0.43 | 0.64 | 0.79 | 0.571 |
| 1500 / hybrid | 0.64 | 0.86 | 0.86 | 0.726 |
