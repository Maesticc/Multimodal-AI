"""Retrieval and answer evaluation against the hand-labelled set in examples/questions.json.

Retrieval metrics (answerable questions only, page-level relevance):
- Hit@k: at least one of the top-k chunks comes from a labelled (doc_id, page).
- MRR@k: mean of 1/rank of the first relevant chunk (0 if none in top-k).
For cross-document questions with `require_all_docs`, a hit requires every
labelled document to appear in the top-k.

Answer checks (used by `examples`): an answer "contains the key facts" if it
contains at least one string from every group in `answer_must_contain`.
This is a cheap string check, not a semantic judgement; see README.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .retrieve import RetrievedChunk, Retriever

QUESTIONS = PROJECT_ROOT / "examples" / "questions.json"
OUT_DIR = PROJECT_ROOT / "examples" / "outputs"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


def _is_relevant(rc: RetrievedChunk, rel: dict) -> bool:
    if rc.chunk.doc_id != rel["doc_id"]:
        return False
    return "page" not in rel or rc.chunk.page == rel["page"]


def first_relevant_rank(results: list[RetrievedChunk], q: dict) -> int | None:
    rel = q["relevant"]
    if q.get("require_all_docs"):
        needed = {r["doc_id"] for r in rel}
        seen: set[str] = set()
        for rc in results:
            seen.add(rc.chunk.doc_id)
            if needed <= seen:
                return rc.rank
        return None
    for rc in results:
        if any(_is_relevant(rc, r) for r in rel):
            return rc.rank
    return None


def answer_has_facts(answer: str, groups: list[list[str]]) -> bool | None:
    if not groups:
        return None
    a = answer.lower()
    return all(any(s.lower() in a for s in g) for g in groups)


def is_evidence(chunk, q: dict) -> bool:
    """True if `chunk` is from a labelled document and its text contains the answer strings."""
    groups = q.get("evidence_must_contain")
    if not groups or chunk.doc_id not in {r["doc_id"] for r in q["relevant"]}:
        return False
    t = chunk.text.lower()
    return all(any(s.lower() in t for s in g) for g in groups)


def evidence_rank(results: list[RetrievedChunk], q: dict) -> int | None:
    """Rank of the first chunk whose text actually contains the answer (stricter than page-level)."""
    return next((rc.rank for rc in results if is_evidence(rc.chunk, q)), None)


def _summarize(ranks: list[int | None], ks) -> dict:
    n = len(ranks)
    s = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    s[f"mrr@{max(ks)}"] = sum(1 / r for r in ranks if r) / n
    return s


def evaluate_retrieval(
    retriever: Retriever, questions: list[dict], mode: str, ks=(1, 3, 5), level: str = "page"
) -> dict:
    """level='page': hit if a chunk from a labelled page is retrieved.
    level='chunk': hit only if a retrieved chunk contains the answer strings in
    `evidence_must_contain` (questions without them are skipped)."""
    kmax = max(ks)
    per_q = []
    for q in questions:
        if not q["relevant"] or (level == "chunk" and not q.get("evidence_must_contain")):
            continue
        res = retriever.retrieve(q["question"], top_k=kmax, mode=mode)
        rank = evidence_rank(res, q) if level == "chunk" else first_relevant_rank(res, q)
        per_q.append({"id": q["id"], "type": q["type"], "rank": rank})
    summary = _summarize([p["rank"] for p in per_q], ks)
    return {"mode": mode, "level": level, "n": len(per_q), "summary": summary, "per_question": per_q}


def format_table(results: list[dict], ks=(1, 3, 5), label: str = "mode") -> str:
    kmax = max(ks)
    head = f"| {label} | " + " | ".join(f"Hit@{k}" for k in ks) + f" | MRR@{kmax} |"
    lines = [head, "|" + "---|" * (len(ks) + 2)]
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r.get('label', r['mode'])} | " + " | ".join(f"{s[f'hit@{k}']:.2f}" for k in ks) + f" | {s[f'mrr@{kmax}']:.3f} |"
        )
    return "\n".join(lines)


def run_eval(settings: Settings, ks=(1, 3, 5), out_name: str = "retrieval_eval") -> list[dict]:
    questions = load_questions()
    retriever = Retriever(settings)
    modes = ("dense", "bm25", "hybrid")
    page = [evaluate_retrieval(retriever, questions, m, ks, "page") for m in modes]
    chunk = [evaluate_retrieval(retriever, questions, m, ks, "chunk") for m in modes]

    def per_q_table(results: list[dict]) -> list[str]:
        lines = ["| id | type | " + " | ".join(r["mode"] for r in results) + " |", "|---|---|" + "---|" * len(results)]
        for i, p in enumerate(results[0]["per_question"]):
            cells = [str(r["per_question"][i]["rank"] or "miss") for r in results]
            lines.append(f"| {p['id']} | {p['type']} | " + " | ".join(cells) + " |")
        return lines

    md = [
        "# Retrieval evaluation",
        "",
        f"Labelled questions from `examples/questions.json`. chunk_size={settings.chunk_size}, "
        f"overlap={settings.chunk_overlap}, embed={settings.embed_model}, candidate_k={settings.candidate_k}, "
        f"rrf_k={settings.rrf_k}.",
        "",
        f"## Page-level ({page[0]['n']} questions)",
        "",
        "Hit if any retrieved chunk comes from a page that contains the answer.",
        "",
        format_table(page, ks),
        "",
        f"## Chunk-level ({chunk[0]['n']} questions)",
        "",
        "Hit only if a retrieved chunk *itself* contains the answer strings (`evidence_must_contain`). "
        "This is what the LLM actually needs; page-level can count a neighbouring chunk on the same page as a hit.",
        "",
        format_table(chunk, ks),
        "",
        "## Per question: rank of first evidence chunk (top-5, `miss` = not retrieved)",
        "",
        *per_q_table(chunk),
        "",
        "## Per question: rank of first chunk from a relevant page",
        "",
        *per_q_table(page),
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{out_name}.md").write_text("\n".join(md), encoding="utf-8")
    (OUT_DIR / f"{out_name}.json").write_text(json.dumps({"page": page, "chunk": chunk}, indent=2), encoding="utf-8")
    print("\n".join(md))
    return page + chunk
