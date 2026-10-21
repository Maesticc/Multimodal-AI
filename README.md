# RAG for Multimodal and Embodied AI Research Papers

A local Retrieval-Augmented Generation (RAG) system for asking questions
about four research papers. Each answer includes citations that point to
the source paper, PDF page, and section.

The system runs locally with Ollama for text generation,
`sentence-transformers` for embeddings, and ChromaDB for vector storage.
After the required models and papers have been downloaded, questions can
be processed without external API keys.

## Papers in the collection

| Document ID | Paper | Venue | Pages | Chunks |
|---|---|---|---:|---:|
| `mmvp` | *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs* — Tong et al. | CVPR 2024 | 11 | 54 |
| `hallusionbench` | *HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in LVLMs* — Guan et al. | CVPR 2024 | 11 | 55 |
| `rt2` | *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Brohan et al. | CoRL 2023, arXiv v1 | 26 | 90 |
| `diffusion_policy` | *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — Chi et al. | RSS 2023 | 16 | 102 |
| **Total** | **4 PDFs** | | **64** | **301** |

The papers cover two related areas: multimodal model evaluation (MMVP
and HallusionBench) and embodied AI policies (RT-2 and Diffusion
Policy). Their overlapping terminology makes cross-paper retrieval
useful to test.

The source URLs and versions are listed in
[`data/sources.json`](data/sources.json). PDF files are not committed to
the repository; they can be downloaded with `scripts/download_data.py`.
The indexed copies were checked against the downloaded PDFs by comparing
page counts and extracted text at the start of each page.

The PDFs contain selectable text, so OCR is not required. However,
tables, figures, and equations are not preserved as structured content
during extraction. See [Known limitations](#known-limitations).

## How it works

The pipeline has two stages: indexing the papers and answering
questions.

### Indexing

1. Read each PDF page with `pypdf`.
2. Detect section headings and attach section metadata to the extracted
   text.
3. Split text into chunks of about 1,000 characters, with roughly 200
   characters of overlap. Chunks do not cross page boundaries so each
   citation maps to one PDF page.
4. Create embeddings with `BAAI/bge-small-en-v1.5`.
5. Store the vectors and metadata in ChromaDB, and prepare the BM25
   index.

### Question answering

1. Encode the question and retrieve candidates with both dense search
   and BM25.
2. Combine the two ranked lists with Reciprocal Rank Fusion (RRF).
3. Send the top five passages, with their paper titles, page numbers,
   and section names, to the local LLM.
4. Generate a short answer with inline citation markers such as `[1]`.
5. Match those markers back to the passages supplied to the model, then
   return the source metadata.

```text
PDFs
  └─ pypdf page extraction
       └─ section detection and chunking
            ├─ BM25 index
            └─ BGE embeddings → ChromaDB

Question
  ├─ Dense retrieval ─┐
  └─ BM25 retrieval ──┴─ RRF → top passages
                                  └─ Ollama LLM
                                       └─ Answer + source citations
```

## Project structure

| File | Purpose |
|---|---|
| [`rag/config.py`](rag/config.py) | Loads settings from environment variables and `.env`. |
| [`rag/ingest.py`](rag/ingest.py) | Extracts PDF text and creates section-aware chunks with metadata. |
| [`rag/index.py`](rag/index.py) | Creates embeddings, stores them in ChromaDB, and writes the index manifest. |
| [`rag/retrieve.py`](rag/retrieve.py) | Runs dense, BM25, or hybrid retrieval and records ranks and scores. |
| [`rag/generate.py`](rag/generate.py) | Builds the prompt, manages the context budget, calls Ollama, and parses citations. |
| [`rag/pipeline.py`](rag/pipeline.py) | Connects the steps in `ask()`, logs queries, and formats answers as Markdown. |
| [`rag/evaluate.py`](rag/evaluate.py) | Evaluates retrieval against labelled questions. |
| [`rag/examples.py`](rag/examples.py) | Regenerates the saved example outputs. |
| [`rag/cli.py`](rag/cli.py) | Provides the command-line interface. |
| [`app.py`](app.py) | Optional Streamlit interface on top of the same pipeline. |
| [`run_app.bat`](run_app.bat) | Starts `app.py` with the project's `.venv` Python (Windows). |
| [`scripts/download_data.py`](scripts/download_data.py) | Downloads the source PDFs. |
| [`scripts/chunk_sweep.py`](scripts/chunk_sweep.py) | Runs the chunk-size experiment. |

## Retrieval details

### Dense retrieval

Dense retrieval uses
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5),
a 33-million-parameter embedding model that produces 384-dimensional
vectors. The query is given the retrieval instruction recommended by the
model card; document passages are not. ChromaDB stores the vectors
locally and uses cosine distance for similarity search.

### Keyword retrieval with BM25

BM25 uses `rank-bm25` with the Okapi variant. It searches the paper
title, section name, and chunk text. This helps retrieve passages
containing exact terms and identifiers such as `DDIM`, `Pct. Diff`, and
`RT-2-PaLI-X-55B`.

### Hybrid retrieval

The default mode retrieves the top 20 candidates from each retriever and
combines their rankings with Reciprocal Rank Fusion:

`score = Σ 1 / (60 + rank)`

RRF combines ranks rather than raw scores, so the cosine and BM25 scores
do not need to be calibrated against each other. The five highest-ranked
passages are passed to the LLM by default.

The retrieval output includes each candidate's dense rank and cosine
score, BM25 rank and score, and fused score. These details are available
through `--show-context`, the `retrieve` command, and
`logs/queries.jsonl`.

## Text extraction and chunking

- **Page-level extraction:** Each page is processed separately, and
  the original one-based PDF page number is stored in the metadata.
- **Text cleanup:** Unicode text is normalised with NFKC, and words
  split by a line-break hyphen are rejoined where possible.
- **Section detection:** The heading detector handles numbered
  headings, Roman numerals, appendix letters, and headings such as
  `Abstract` and `Limitations`. It filters common false positives and
  repairs some headings extracted with spaced small caps. The detected
  sections were inspected for these four papers only.
- **References:** Text after a `References` heading is skipped until
  an appendix heading is found. Set `RAG_INCLUDE_REFERENCES=true` to
  include references.
- **Chunk size:** Sentences are packed into chunks of about 1,000
  characters with about 200 characters of overlap. A chunk stays
  within one page. A section change starts a new chunk once the
  current chunk has at least 300 characters, and fragments shorter
  than 50 characters are dropped.
- **Metadata:** Each chunk stores `chunk_id`, `doc_id`, title,
  filename, page, section, and text. For example, `rt2:p5:c23`
  identifies a chunk from document `rt2`, page 5.

## Answer generation and citations

The generation prompt instructs the model to:

- Use only the retrieved passages to answer.
- Add a citation marker such as `[1]` after each supported claim.
- Keep answers to six sentences or fewer.
- Copy numerical values exactly.
- Respond with `I cannot answer this from the provided documents.`
  when the passages do not contain enough evidence.

Each passage is labelled with its title, page, and section. The context
budget is 6,000 characters, approximately 1,500 tokens. Passages are
removed from the lowest-ranked end if the budget is exceeded; a passage
is only truncated if it alone is longer than the budget. The generation
context uses `num_ctx=4096`, leaving room for the system prompt,
question, and an answer of up to 512 tokens. In the saved runs, all five
passages fit within the budget.

Generation uses temperature `0` and seed `42`. Citation markers are
mapped back to the passages included in the prompt. Each source record
contains the title, filename, page, section, and chunk ID. Citation
numbers that do not match a supplied passage are recorded as
`invalid_citations` rather than being assigned to a source.

## Models and dependencies

| Component | Version | Reason for use |
|---|---|---|
| LLM | `llama3.2:3b` through Ollama 0.40.6; Q4_K_M quantisation | Runs locally and fits the project's 4 GB VRAM GPU. |
| Embeddings | `BAAI/bge-small-en-v1.5` with `sentence-transformers` 6.1.0 | Small English retrieval model; runs on CPU so it does not compete with the LLM for GPU memory. |
| Vector store | ChromaDB 1.5.9 | Local persistent storage for vectors and metadata. |
| Sparse retrieval | `rank-bm25` 0.2.2 | Keyword-based ranking for exact terms. |
| PDF extraction | `pypdf` 6.21.1 | Extracts text one page at a time. |

The tested setup uses Python 3.13.12 on Windows 11 and an NVIDIA RTX
3050 Laptop GPU with 4 GB VRAM. Ollama reports about 2.9 GB loaded, with
roughly 80% GPU and 20% CPU use. Once the model is loaded, a question
typically takes around 4–8 seconds; the first question after a cold
start took about 28 seconds in the recorded run. A larger 7B–8B model
would be more likely to spill onto the CPU with this GPU.

A GPU is optional. The system can run on CPU, but generation will be
slower and may take tens of seconds per question. Allow around 3 GB of
disk space for the LLM.

The LLM can be changed through `RAG_LLM_MODEL` (for example,
`qwen2.5:3b`) without changing the application code. The results below
were produced using the versions listed in this section.

## Setup

1. Create a Python virtual environment and install the pinned
   dependencies:

   ```bash
   python -m venv .venv
   ```

   Activate the environment, then run:

   ```bash
   pip install -r requirements.txt
   ```

2. Install [Ollama](https://ollama.com/download), then download the
   model:

   ```bash
   ollama pull llama3.2:3b
   ```

3. Add the four PDFs to `data/`, or download them using the project
   script:

   ```bash
   python scripts/download_data.py
   ```

4. Optionally copy [`.env.example`](.env.example) to `.env` and adjust
   the settings. The defaults can be used without this file.

5. Build the index:

   ```bash
   python -m rag ingest
   ```

The first indexing run downloads the embedding model if it is not
already available. Indexing the 301 chunks took about two minutes on CPU
in the tested environment.

## Usage

Run the package as a Python module:

| Command | Description |
|---|---|
| `python -m rag ingest` | Builds the vector index from `data/*.pdf`. |
| `python -m rag ask "your question"` | Answers one question and includes source citations. |
| `python -m rag retrieve "your question"` | Shows retrieved passages without calling the LLM. |
| `python -m rag chat` | Starts an interactive question loop. |
| `python -m rag info` | Displays settings, index manifest, and model details. |
| `python -m rag eval` | Evaluates dense, BM25, and hybrid retrieval. |
| `python -m rag examples` | Regenerates `examples/outputs/answers.md`. |

The `ask` command supports options for displaying retrieved context,
selecting a retrieval mode, changing `top_k`, and printing the full JSON
record. Run a command with `--help` to see the available options.

The chunk-size experiment can be run with:

```bash
python scripts/chunk_sweep.py
```

Each question is appended to `logs/queries.jsonl`, including the prompt,
retrieved passages and their scores, token counts, and latency.
Configuration options are documented in [`.env.example`](.env.example),
including chunk size and overlap, embedding model and device, retrieval
mode, candidate counts, RRF settings, Ollama URL, temperature, seed,
context length, character budget, and answer token limit.

### Web interface (optional)

`app.py` is a Streamlit interface that calls the same
`RAGPipeline.ask()` as the CLI. It shows the answer, cited sources,
retrieved passages with their scores, and the prompt. A *Saved
examples* view browses the 17 recorded runs and works without Ollama.

On Windows, double-click `run_app.bat`, or run it from a terminal. It
starts the app with the project's `.venv` Python. You can also start it
directly:

```bash
.venv/Scripts/python.exe -m streamlit run app.py --server.address localhost
```

Use the `.venv` Python rather than a bare `streamlit` command. If
another Python installation (such as Miniconda) has its own Streamlit
on the `PATH`, a bare `streamlit run app.py` starts that one instead,
and the app fails with `ModuleNotFoundError: No module named
'rank_bm25'`.

The app is meant for local demos. It has no authentication, so bind it
to `localhost` as shown above.

## Evaluation

Saved outputs are available in [`examples/outputs/`](examples/outputs/):

- [`answers.md`](examples/outputs/answers.md) — 17 questions,
  answers, citations, and retrieved passages with ranks, scores, and
  text.
- [`manual_review.md`](examples/outputs/manual_review.md) — manual
  checks of the answers against the retrieved evidence.
- [`retrieval_eval.md`](examples/outputs/retrieval_eval.md) —
  retrieval metrics.
- [`chunk_sweep.md`](examples/outputs/chunk_sweep.md) — results from
  the chunk-size experiment.
- `answers.jsonl` — complete records, including prompts.

### Test questions

[`examples/questions.json`](examples/questions.json) contains 17
hand-labelled questions: nine factual, four conceptual, one
paraphrased, one cross-document, and two unanswerable. For answerable
questions, the evaluation records the pages containing the answer and
the evidence strings needed to verify it. A chunk counts as evidence
only when its own text contains those strings.

The page labels were checked across the full set of chunks because some
facts appear on more than one page. For example, `46.9%` appears on
pages 1, 2, 6, 7, and 15 of the Diffusion Policy paper. The labels were
updated after this check.

### Retrieval results

The default index uses 1,000-character chunks with 200 characters of
overlap.

| Mode | Page Hit@1 | Page Hit@3 | Page Hit@5 | Page MRR@5 | Chunk Hit@1 | Chunk Hit@3 | Chunk Hit@5 | Chunk MRR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.60 | 0.87 | 0.93 | 0.724 | **0.57** | **0.86** | **0.86** | **0.714** |
| BM25 | 0.53 | 0.93 | 1.00 | 0.717 | 0.50 | 0.71 | 0.79 | 0.613 |
| Hybrid (default) | 0.53 | **1.00** | **1.00** | **0.744** | 0.43 | 0.79 | 0.86 | 0.621 |

Page-level metrics count a result as a hit if it comes from a page
containing the answer. Chunk-level metrics require the retrieved chunk
itself to contain the answer. Chunk-level retrieval is more directly
relevant to generation because the LLM only receives the retrieved
passages.

The results do not establish that one retrieval mode is universally
better. Hybrid retrieval achieved the best page-level Hit@3 and Hit@5,
while dense retrieval matched or outperformed it across the reported
chunk-level metrics. BM25 can rank a nearby chunk from the correct page
above the chunk containing the actual answer, and RRF can carry that
ranking into the combined results. Hybrid remains the default because it
retrieved the relevant page for every evaluated question at Hit@3 and is
useful for exact-identifier queries. The evaluation set is small: one
question changes Hit@k by roughly 0.07.

Both chunk-level misses (`q01` and `q14`) involve content dominated by
table or figure text. For `q14`, the RoboMimic chunk is mostly flattened
task-table content and ranked 20th with hybrid retrieval and 55th with
dense retrieval (measured with a 60-candidate pool; with the default 20
candidates per retriever, dense retrieval does not return it at all). Raising `top_k` to 10 recovered `q01` at rank 9, but
did not recover `q14`.

### Answer review

The manual review covered all 17 questions.

| Outcome | Questions |
|---|---|
| Correct and grounded | `q02`, `q04`, `q05`, `q06`, `q07`, `q08`, `q10`, `q11`, `q12`, `q13` (10 of 15 answerable questions) |
| Correct facts, one citation attributed incorrectly | `q09` |
| Retrieval failure: evidence was not in the top five, so the model said it was not stated | `q01`, `q14` |
| Generation failure despite evidence being present | `q03` (mixed up Table 2 and Table 3 and added an incorrect `6.5%`); `q15` (misread the statement about models scoring below random) |
| Unanswerable | `q17` correctly abstained; `q16` incorrectly added `batch size 8` from an unrelated hyperparameter table after abstaining on the learning rate |

The system works best for direct lookups where the answer appears in
prose, including counts, percentages, latency figures, named metrics,
definitions, and stated limitations. Answers are short and generally
cite the correct page.

The main problems are table- and figure-based facts, statements
involving negation or scope (such as "except"), partial abstention, and
citation markers copied from the papers' own references. The automatic
string check accepted or partly accepted some incorrect answers. Manual
review caught the extra claim in `q03`, the interpretation error in
`q15`, and the unsupported batch-size claim in `q16`. A check that only
looks for the right number measures whether an answer contains expected
text; it does not establish that the answer is faithful to the source.

### Chunk-size experiment

The index was rebuilt with chunk sizes of 500, 1,000, and 1,500
characters while keeping the other settings fixed. The overlap was set
to 20% of each chunk size.

| Chunk size | Number of chunks | Dense Chunk Hit@5 | Hybrid Chunk Hit@5 | Hybrid Chunk MRR@5 | Hybrid Page Hit@1 |
|---:|---:|---:|---:|---:|---:|
| 500 | 540 | 0.57 | 0.79 | 0.556 | 0.80 |
| 1,000 | 301 | 0.86 | 0.86 | 0.621 | 0.53 |
| 1,500 | 224 | 0.93 | 0.86 | 0.726 | 0.73 |

Chunks of 500 characters performed worse on chunk-level retrieval, even
though their page-level Hit@1 was higher. Small chunks sometimes
separated facts needed to answer one question, such as "346 images" and
"1129 questions." This is another example of why page-level metrics
alone can be misleading.

The difference between 1,000 and 1,500 characters is less conclusive.
The changes amount to roughly one question on several metrics, and
larger passages use more of the available context budget. Five full-size
passages can exceed 6,000 characters, causing lower-ranked passages to
be dropped. The default remains 1,000 characters. Generation was not
rerun with 1,500-character chunks, so this experiment does not show
whether that size improves answer quality.

## Design choices

- **Keep chunks within page boundaries.** This makes page citations
  precise, but can split a sentence at a page break and produce
  shorter chunks.
- **Use character counts for chunking.** This keeps chunk sizing
  simple and independent of a specific tokenizer. For these papers,
  1,000 characters is roughly 250 tokens.
- **Combine rankings with RRF.** No score calibration is needed,
  though BM25 can promote a neighbouring chunk over the chunk that
  contains the answer.
- **Skip reranking and query rewriting for now.** The current pipeline
  provides a baseline that is straightforward to inspect. Both are
  possible next steps.
- **Use a 3B model.** The tested GPU has 4 GB of VRAM. The evaluation
  points to reading precision—especially tables and negation—as a
  major source of errors. A larger model might help, but that has not
  been tested.
- **Use prompt-based abstention.** There is no retrieval-score
  threshold because the score distributions for answerable and
  unanswerable questions overlap in this small test set. In `q16`, for
  example, the top passage looked relevant but contained
  hyperparameters for a different setting.

## Known limitations

- **Tables and figures:** They are extracted as flattened text, for
  example `Lift 1 1 7 200 300 400 Yes No ...`. These chunks are
  difficult to retrieve reliably, and the 3B model may misread them.
  Information contained only in plots or diagrams is not available to
  the system.
- **Extraction noise:** Some icon glyphs appear as text (such as
  `/sync` and `♂search` in the MMVP pattern list), and some words in
  RT-2 lose their spaces (for example, `althoughweshowthat`).
- **Heading detection:** The heading rules were checked on these four
  papers. Other PDF layouts may require changes to the detection
  patterns.
- **Small evaluation set:** There are 15 answerable questions with a
  single set of labels and no independent annotator. The metrics are useful for identifying
  possible trends, but the differences are not statistically
  conclusive.
- **Manual answer review:** Answer correctness was judged in a single
  manual review pass, without a second reviewer. The automatic evaluation uses string matching and cannot
  reliably detect unsupported claims.
- **English and single-turn use:** The current interface is an
  English-language command-line tool without conversation memory.

## Next steps

1. **Improve table extraction.** Extract tables separately with a tool
   such as `pdfplumber` or a layout model. Store them as separate
   chunks with descriptions based on their captions, then measure
   chunk-level Hit@5 on `q01`, `q14`, and additional table-focused
   questions.
2. **Add a reranker.** Test a cross-encoder such as `bge-reranker-base`
   on the hybrid top 20 to see whether it improves the first-stage
   ranking.
3. **Check answer faithfulness.** Evaluate claims against their
   supporting passages with NLI or an LLM-based judge, and compare the
   results with manual labels.
4. **Compare generation models.** Keep retrieval fixed and compare
   `llama3.2:3b`, `qwen2.5:3b`, and an 8B model running on CPU. This
   would help separate retrieval problems from generation problems.
5. **Expand the evaluation set.** Add more independently written
   questions so that small differences in Hit@k are easier to
   interpret.

## How this project was built

I built this project with help from an AI coding assistant (Kiro). The
assistant drafted most of the code, the evaluation questions and their
labels, the answer review in `manual_review.md`, and the first version
of this README. I chose the task, the papers, and the hardware setup,
directed the changes, and edited the final README. All experiments and
saved outputs were produced by running the code on my machine.

## References

- Tong et al., *Eyes Wide Shut? Exploring the Visual Shortcomings of
  Multimodal LLMs*, CVPR 2024. arXiv:2401.06209.
- Guan et al., *HallusionBench*, CVPR 2024. arXiv:2310.14566.
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web
  Knowledge to Robotic Control*, CoRL 2023. arXiv:2307.15818.
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion*, RSS 2023. arXiv:2303.04137.
- Cormack, Clarke, and Büttcher, "Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods," SIGIR 2009.
- Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese
  Embedding" (BGE models), 2023. arXiv:2309.07597.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25
  and Beyond," 2009.
