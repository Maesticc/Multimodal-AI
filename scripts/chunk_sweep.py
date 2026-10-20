"""Exploration: effect of chunk size on retrieval quality.

Builds a separate index for each chunk size (overlap fixed at 20% of chunk size)
under index_sweep/, evaluates dense / BM25 / hybrid retrieval on the labelled
questions, and writes examples/outputs/chunk_sweep.md.

Usage:  python scripts/chunk_sweep.py [500 1000 1500]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import PROJECT_ROOT, get_settings  # noqa: E402
from rag.evaluate import OUT_DIR, evaluate_retrieval, format_table, load_questions  # noqa: E402
from rag.index import build_index  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    sizes = [int(x) for x in sys.argv[1:]] or [500, 1000, 1500]
    questions = load_questions()
    rows, raw = [], []
    for cs in sizes:
        s = get_settings(chunk_size=cs, chunk_overlap=cs // 5, index_dir=PROJECT_ROOT / "index_sweep" / f"cs{cs}")
        mpath = s.index_dir / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
        if not manifest or manifest["settings"]["chunk_size"] != cs or manifest["settings"]["embed_model"] != s.embed_model:
            manifest = build_index(s)
        retriever = Retriever(s)
        for level in ("page", "chunk"):
            for mode in ("dense", "bm25", "hybrid"):
                r = evaluate_retrieval(retriever, questions, mode, level=level)
                r["label"] = f"{cs} / {mode}"
                r["chunk_size"], r["num_chunks"] = cs, manifest["num_chunks"]
                rows.append(r)
                raw.append(r)
        print(f"chunk_size={cs}: {manifest['num_chunks']} chunks")

    page = [r for r in rows if r["level"] == "page"]
    chunk = [r for r in rows if r["level"] == "chunk"]
    md = [
        "# Chunk-size sweep",
        "",
        "Overlap = 20% of chunk size; all other settings default. Same labelled questions as `retrieval_eval.md`. "
        "Caveats: larger chunks cover more text, so a top-5 of 1500-char chunks shows the LLM ~3x more text than a "
        "top-5 of 500-char chunks (and is more likely to hit the context cap); with only 15 questions, one question "
        "moves Hit@k by 0.067.",
        "",
        "| chunk size | chunks |",
        "|---|---|",
        *[f"| {r['chunk_size']} | {r['num_chunks']} |" for r in page if r["mode"] == "dense"],
        "",
        "## Page-level",
        "",
        format_table(page, label="chunk size / mode"),
        "",
        "## Chunk-level (retrieved chunk contains the answer strings)",
        "",
        format_table(chunk, label="chunk size / mode"),
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chunk_sweep.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / "chunk_sweep.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
