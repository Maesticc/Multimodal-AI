# RAG over Multimodal & Embodied AI Papers

Technical Assessment, Option 1 (RAG Implementation). AI Research Engineer Intern, Multimodal AI.

A local Retrieval-Augmented Generation system that answers questions about four research papers and cites the document, page and section of the evidence it used. It runs fully offline after setup: Ollama provides the LLM, sentence-transformers the embeddings and ChromaDB the vector store. No API keys are required.

## 1. Document collection

| doc_id | Paper | Venue | Pages | Chunks |
|---|---|---|---|---|
| `mmvp` | Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs (Tong et al.) | CVPR 2024 | 11 | 54 |
| `hallusionbench` | HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in LVLMs (Guan et al.) | CVPR 2024 | 11 | 55 |
| `rt2` | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (Brohan et al.) | CoRL 2023, arXiv v1 | 26 | 90 |
| `diffusion_policy` | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al.) | RSS 2023 | 16 | 102 |
| **Total** | 4 PDFs, ~5.5 MB, 64 pages, ~229k characters indexed | | **64** | **301** |

- **Source and permission:** all four papers are on the assessment reading list (P1, P5, P6, P8) and are publicly available from CVF Open Access, arXiv and the project page. URLs and versions are in [`data/sources.json`](data/sources.json). The PDFs are **not committed**; the script `scripts/download_data.py` fetches them. I checked that the downloaded files match the copies I indexed page for page (same page count and the same extracted text at the start of every page), so page citations stay valid.
- **Why this collection:** two multimodal benchmark papers (MMVP and HallusionBench, which both evaluate GPT-4V) and two embodied-AI policy papers (RT-2 and Diffusion Policy). The overlap in vocabulary (diffusion, VLMs, benchmarks, "tokens", "Hz") makes retrieval non-trivial, and it allows cross-document questions.
- **File type:** born-digital PDF (text layer, no OCR needed). Figures, tables and equations come out only as flattened text (see Limitations).

## 2. Architecture

```
            ingest (offline, once)                              query (online)
 ┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────────────┐
 data/*.pdf                                     question
   │ pypdf, page by page                          │
   ▼                                              ├─► dense: bge-small query embedding ─► Chroma (cosine) ─┐ top-20
 page text ─► heading detection (section)         │                                                     ├─► RRF fusion ─► top-5
   │          skip References                     └─► sparse: BM25 over title+section+text ─────────────┘ top-20
   ▼                                                                                    │
 sentence split ─► pack ~1000 chars, 200 overlap,                     numbered context [1]..[5]  (≤ 6000 chars)
                   never across pages                                 + title / page / section headers
   │                                                                                    │
   ▼                                                                                    ▼
 chunks.jsonl (text + metadata) ─► BM25                                llama3.2:3b via Ollama (temp 0, seed 42)
   │                                                                                    │
   └─► bge-small-en-v1.5 embeddings ─► ChromaDB                         answer with [n] ─► parse [n] ─► sources
                                                                                        │   (doc, page, section, chunk_id)
                                                                         logs/queries.jsonl (full prompt + scores)
```

| File | Role |
|---|---|
| [`rag/config.py`](rag/config.py) | Every setting comes from env vars or `.env`, nothing is hard-coded |
| [`rag/ingest.py`](rag/ingest.py) | PDF to page text to section-aware chunks with metadata |
| [`rag/index.py`](rag/index.py) | Embeddings, ChromaDB persistence, `manifest.json` |
| [`rag/retrieve.py`](rag/retrieve.py) | Dense, BM25 and hybrid (RRF) retrieval with per-retriever ranks and scores |
| [`rag/generate.py`](rag/generate.py) | Prompt construction, context budget, Ollama client, citation parsing |
| [`rag/pipeline.py`](rag/pipeline.py) | End-to-end `ask()`, JSONL logging, Markdown rendering |
| [`rag/evaluate.py`](rag/evaluate.py), [`rag/examples.py`](rag/examples.py) | Labelled evaluation and saved example outputs |
| [`rag/cli.py`](rag/cli.py) | Command-line interface |

### 2.1 Ingestion

1. **Text extraction:** `pypdf` reads each page separately, so every chunk keeps a 1-based PDF page number. Ligatures are normalised (NFKC) and words hyphenated across line breaks are re-joined (`rea-\nsoning` becomes `reasoning`).
2. **Section names:** each line is tested against a heading regex: numbered (`2.1. Finding CLIP-blind Pairs`), roman (`IV. INTRIGUING PROPERTIES`), appendix letters (`D. Hyperparameters`) and named headings (`Abstract`, `Limitations`). Filters reject false positives such as list items, affiliations and table rows. Small-caps headings that extract as `I NTRODUCTION` are repaired. The current section carries across pages. Detected sections were inspected by hand for all four papers.
3. **References are skipped:** text after a `References` heading is dropped until an appendix heading appears (`RAG_INCLUDE_REFERENCES=true` to keep it). Bibliographies match many query terms but answer nothing.
4. **Chunking:** sentences are packed greedily into ~1000-character chunks with ~200 characters of sentence-level overlap. A chunk **never crosses a page boundary**, so each one has exactly one page to cite. A section change starts a new chunk once the current chunk has at least 300 characters. Fragments under 50 characters (stray figure labels) are dropped.
5. **Metadata per chunk:** `chunk_id` (`rt2:p5:c23` = document, page, running index), `doc_id`, title, file, page, section and text.

### 2.2 Retrieval

- **Dense:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (33M params, 384-dim, normalised). Queries get the BGE retrieval instruction prefix and passages do not, as the model card recommends. ChromaDB with HNSW and cosine distance.
- **Sparse:** BM25 (`rank-bm25`, Okapi) over `title + section + text`, with a small stop-word list. It catches exact identifiers such as `DDIM`, `Pct. Diff` and `RT-2-PaLI-X-55B`.
- **Hybrid (default):** top-20 from each retriever, fused with Reciprocal Rank Fusion, `score = Σ 1/(60 + rank)`. RRF uses ranks only, so cosine and BM25 scores never need to be calibrated against each other.
- **Settings:** `top_k=5` passages go to the LLM. Every result records its dense rank and cosine score, its BM25 rank and score, and the fused score. They appear in `--show-context`, `retrieve`, `logs/queries.jsonl` and `examples/outputs/answers.md`.

### 2.3 Generation and context limits

- The system prompt (in [`generate.py`](rag/generate.py)) tells the model to use **only** the numbered passages, cite `[n]` after each claim, answer in at most 6 sentences, copy numbers exactly, and reply with the fixed sentence `I cannot answer this from the provided documents.` when the evidence is insufficient.
- Each passage is shown as `[n] (source: <title> | page <p> | section: <s>)` followed by its text, in retrieval order.
- **Context budget:** at most 6000 characters of passages (~1500 tokens). Whole passages are dropped from the lowest-ranked end, never cut mid-passage. Only the top passage may be truncated, and only if it alone exceeds the budget. With `num_ctx=4096` this leaves room for the system prompt, the question and a 512-token answer. In the saved runs all 5 passages always fit (4,360–5,584 context characters, 1,128–1,641 prompt tokens).
- **Deterministic decoding:** temperature 0 and seed 42.

### 2.4 Source attribution

The `[n]` markers in the answer are parsed back to the passages that were actually in the prompt, giving `title, file, page, section, chunk_id` for each one. Sources are therefore derived from the evidence the model saw, not generated by the model. Numbers that match no passage (for example, the 3B model copying a paper's own `[12, 15, 29]` references) are reported as `invalid_citations` instead of being silently mapped.

## 3. Models and why

| Component | Exact version | Why |
|---|---|---|
| LLM | `llama3.2:3b` on Ollama 0.40.6 (3.2B params, Q4_K_M, digest `a80c4f17acd5`) | Free and local. It fits the 4 GB VRAM of my RTX 3050 laptop GPU (Ollama reports 2.9 GB loaded, 80% GPU / 20% CPU). It follows instructions well enough for cited answers at ~4–8 s per question once loaded (the first query after a cold start took ~28 s). A 7–8B model would mostly spill to CPU on this card. |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers 6.1.0 | Strong small English retrieval model. It runs on CPU so it does not compete with the LLM for the 4 GB of VRAM, and indexing 301 chunks takes ~2 minutes on CPU. |
| Vector store | ChromaDB 1.5.9 (persistent, local) | Free, file-based, no server, and stores metadata next to the vectors. |
| Sparse | rank-bm25 0.2.2 | Exact-term matching that the small dense model misses. |
| PDF | pypdf 6.21.1 | Pure Python, page-level extraction. |

Any Ollama model can be used by changing `RAG_LLM_MODEL` (for example `qwen2.5:3b`), with no code change. The saved outputs were produced with the settings above.

## 4. Setup

**Requirements:** Python 3.11+ (tested on 3.13.12, Windows 11), ~3 GB disk for the model, ~4 GB RAM. A GPU is optional: on CPU only, Ollama still works, just slower (expect tens of seconds per answer). Tested on an NVIDIA RTX 3050 Laptop GPU with 4 GB.

Setup takes five steps:

1. Create a Python virtual environment and install the pinned packages from [`requirements.txt`](requirements.txt).
2. Install Ollama from [ollama.com/download](https://ollama.com/download) and pull the `llama3.2:3b` model.
3. Put the four PDFs in `data/`, or let `scripts/download_data.py` download them.
4. Optionally copy [`.env.example`](.env.example) to `.env` to change settings. The defaults work without it.
5. Build the index with the `ingest` command of the `rag` package. It takes ~2 minutes on CPU, and the embedding model is downloaded on the first run.

## 5. Usage

The `rag` package is run as a Python module and has these commands:

| Command | What it does |
|---|---|
| `ingest` | Builds the vector index from `data/*.pdf` |
| `ask` | Answers one question with cited sources. Options can show the retrieved passages, switch the retrieval mode, change `top_k` or print the full JSON record |
| `retrieve` | Retrieval only, without the LLM |
| `chat` | Interactive question loop |
| `info` | Shows the settings, the index manifest and the exact model and digest |
| `eval` | Retrieval evaluation: dense vs BM25 vs hybrid |
| `examples` | Regenerates `examples/outputs/answers.md` |

The chunk-size experiment is in `scripts/chunk_sweep.py`.

Every question is appended to `logs/queries.jsonl`, together with the full prompt, all retrieved passages and their scores, and the token counts and latency.

Configuration lives in [`.env.example`](.env.example). It covers paths, chunk size and overlap, embedding model and device, retrieval mode, `top_k`, `candidate_k`, `rrf_k`, the LLM provider and model, the Ollama URL, temperature, seed, `num_ctx`, the context character budget and the answer token limit.

## 6. Results

All outputs are saved in [`examples/outputs/`](examples/outputs/), so the system's behaviour can be reviewed without running it:

- [`answers.md`](examples/outputs/answers.md): 17 questions, each with its answer, cited sources and the retrieved passages (ranks, scores, text).
- [`manual_review.md`](examples/outputs/manual_review.md): my check of every answer against its passages.
- [`retrieval_eval.md`](examples/outputs/retrieval_eval.md) and [`chunk_sweep.md`](examples/outputs/chunk_sweep.md): retrieval metrics.
- `answers.jsonl`: full records, including prompts.

### 6.1 Evaluation set

[`examples/questions.json`](examples/questions.json) has 17 hand-written questions: 9 factual, 4 conceptual, 1 paraphrased (it avoids the paper's wording), 1 cross-document and 2 unanswerable. For each answerable question I labelled:

- the **pages** that contain the answer, and
- **evidence strings** (for example `0.95` and `0.6`). A retrieved chunk counts as evidence only if its own text contains them. This label is independent of the chunk ids, so it works for any chunk size.

Labelling note: my first page labels were incomplete. Several facts are restated elsewhere in the paper (46.9% appears on pages 1, 2, 6, 7 and 15 of Diffusion Policy). I found this by searching every chunk for the answer strings, and fixed the labels before reporting any of the numbers below.

### 6.2 Retrieval (default index: chunk 1000 / overlap 200)

| | Page-level (15 q) | | | | Chunk-level (14 q) | | | |
|---|---|---|---|---|---|---|---|---|
| mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| dense | 0.60 | 0.87 | 0.93 | 0.724 | **0.57** | **0.86** | **0.86** | **0.714** |
| BM25 | 0.53 | 0.93 | 1.00 | 0.717 | 0.50 | 0.71 | 0.79 | 0.613 |
| hybrid (default) | 0.53 | **1.00** | **1.00** | **0.744** | 0.43 | 0.79 | 0.86 | 0.621 |

*Page-level:* a retrieved chunk comes from a page that contains the answer. *Chunk-level:* a retrieved chunk itself contains the answer. Chunk-level is what the LLM actually needs.

What this shows:

- **The metric changes the conclusion.** At page level, hybrid looks best (Hit@3 = 1.00). At chunk level, dense alone is as good or better on every column. BM25 often ranks a *neighbouring* chunk on the right page (same rare terms) above the chunk that actually holds the number, and RRF inherits that. I kept hybrid as the default because it never misses the relevant page and it is more robust to exact-identifier queries. With 14–15 questions, though, one question moves Hit@k by ~0.07, so I would not claim either mode is better from this data.
- **Both chunk-level misses (q01, q14) are the same failure type:** the answer sits in a chunk dominated by table or figure text. For q14, the Robomimic chunk is mostly a flattened task table and ranks 20th in hybrid and 55th in dense. Raising `top_k` to 10 recovers q01 (rank 9) but not q14.

### 6.3 End-to-end answers (manual review, 17 questions)

| Outcome | Questions |
|---|---|
| Correct and grounded | q02, q04, q05, q06, q07, q08, q10, q11, q12, q13 (10/15) |
| Correct facts, one mis-attributed citation | q09 |
| **Retrieval failure:** evidence chunk not in top-5, model correctly said it was not stated | q01, q14 |
| **Generation failure:** evidence was in the prompt | q03 (mixed up Table 2 and Table 3: an extra wrong "6.5%"), q15 (read "models *except* GPT-4V scored below random" as GPT-4V scoring below random) |
| Unanswerable | q17 abstained ✓. q16 ✗: abstained on the learning rate, then invented "batch size 8" from an unrelated hyperparameter table |

**Works well:** single-fact lookups where the fact sits in prose (counts, percentages, latencies, named metrics), definitions and stated limitations. Answers are short, cite the right page, and copy numbers exactly.

**Struggles:** (1) facts that live in tables or figures, at both retrieval and reading time; (2) passages with negation or scope ("except", "smaller version of"); (3) partial abstention: the model sees *a* learning rate and batch size and attaches them to the wrong model; (4) it sometimes copies the paper's own reference brackets as citations.

The automatic string check in `answers.md` marked q03, q15 and q16's partial answer as acceptable or only partly wrong. Only the manual review caught the wrong extra claims. A cheap "contains the right number" check measures recall, not faithfulness.

## 7. Exploration: chunk size

**Question:** does chunk size explain the table-heavy misses, and is 1000 characters a reasonable default?
**Method:** rebuilt the index at 500, 1000 and 1500 characters (overlap 20%), everything else fixed, and measured both metrics on the same questions ([`chunk_sweep.md`](examples/outputs/chunk_sweep.md)).

| chunk size | chunks | dense chunk-Hit@5 | hybrid chunk-Hit@5 | hybrid chunk-MRR@5 | hybrid page-Hit@1 |
|---|---|---|---|---|---|
| 500 | 540 | 0.57 | 0.79 | 0.556 | 0.80 |
| 1000 | 301 | 0.86 | 0.86 | 0.621 | 0.53 |
| 1500 | 224 | 0.93 | 0.86 | 0.726 | 0.73 |

**Findings:**

- **500 characters is clearly worse at chunk level** (dense 0.57 vs 0.86) even though its page-level Hit@1 is the highest. Small chunks often split a question's two facts (for example "346 images" and "1129 questions") across chunks, and they find the right page without the full answer. This is the clearest case where page-level metrics mislead.
- **1500 versus 1000 is a small difference, not a clear win.** It is about one question on several metrics. It also sends ~1.5× more text per passage, so five full-size passages can exceed the 6000-character budget, in which case the lowest-ranked one is dropped.
- I kept **1000** as the default. I did not re-run generation at 1500, so I have **no evidence about answer quality** at that size.

## 8. Design decisions and trade-offs

- **Page-bounded chunks:** citations are always exactly one page. The cost is that sentences spanning a page break are split and some chunks are short.
- **Character-based sizes instead of tokens:** simple and model-independent. 1000 characters is roughly 250 tokens for this text.
- **Hybrid with RRF instead of a weighted score sum:** no score calibration needed. The trade-off, measured in 6.2, is that BM25 can promote neighbouring chunks.
- **No reranker or query rewriting:** I wanted a baseline whose every step I can trace and explain. These are listed as next steps instead.
- **3B LLM:** constrained by 4 GB of VRAM. Section 6.3 shows its failure modes are mostly reading precision (tables, negation), which is expected to improve with a larger model, but I have not tested that.
- **Abstention is prompt-only:** no retrieval-score threshold, because the score distributions for answerable and unanswerable questions overlap on this small set (q16's top passage is a genuinely relevant-looking hyperparameter table).

## 9. Known limitations

- **Tables and figures** are extracted as flattened text (`Lift 1 1 7 200 300 400 Yes No ...`). These chunks embed poorly and the 3B model reads them unreliably. Figure-only content (plots, diagrams) is invisible to the system. This is the main gap for a *multimodal* paper collection.
- **Extraction noise:** icon glyphs appear as text (`/sync`, `♂search` in the MMVP pattern list), and some words in RT-2 lose their spaces (`althoughweshowthat`).
- **Heading detection is heuristic.** It was checked by eye on these four papers, and other layouts may need regex changes.
- **Small evaluation set** (15 answerable questions, labelled by me alone). The metrics show tendencies and are not statistically meaningful differences.
- **Answer correctness was judged manually by one person.** The automatic check is string matching only.
- **English only,** and a single-turn CLI with no conversation memory.

## 10. What I would investigate next

1. **Table-aware ingestion,** since it is the cause of both retrieval misses. Extract tables separately (for example with `pdfplumber` or a layout model), store them as their own chunks with a caption-based description, and measure chunk-level Hit@5 on q01 and q14 plus new table-specific questions.
2. **A cross-encoder reranker** (for example `bge-reranker-base`) over the hybrid top-20. This tests whether the "right page, wrong chunk" errors in 6.2 come from first-stage ranking.
3. **A faithfulness check:** an automatic claim-versus-passage verification (NLI or an LLM judge), validated against my manual labels, to catch q03/q15-style errors that the string check misses.
4. **A model comparison** with retrieval held fixed (`llama3.2:3b` vs `qwen2.5:3b` vs an 8B model on CPU). This tests whether the generation errors are about model size, so retrieval and generation effects can be separated cleanly.
5. **A larger, independently written question set,** so that differences of 0.07 become meaningful.

## 11. Use of AI assistance

> **TODO (author):** edit this section so it accurately describes your own process before you submit.

This project was built with an AI coding assistant (Kiro). The assistant drafted most of the code, the evaluation scripts and this README, based on my requirements and the assessment brief. I reviewed and ran the pipeline, and the results reported here come from those runs. <!-- Describe what you checked, changed or wrote yourself, e.g. reading every answer against its passages, verifying the question labels, choosing the default settings. -->

## References

- Tong et al., *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*, CVPR 2024. arXiv:2401.06209
- Guan et al., *HallusionBench*, CVPR 2024. arXiv:2310.14566
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL 2023. arXiv:2307.15818
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. arXiv:2303.04137
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009.
- Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE models), 2023. arXiv:2309.07597
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
