"""End-to-end RAG pipeline: question -> retrieve -> build prompt -> generate -> cite -> log."""

from __future__ import annotations

import json
import time
from datetime import datetime

from .config import Settings
from .generate import ABSTAIN, SYSTEM_PROMPT, build_context, build_user_prompt, get_llm, parse_citations
from .retrieve import Retriever


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.retriever = Retriever(settings)
        self.llm = get_llm(settings)

    def ask(self, question: str, top_k: int | None = None, mode: str | None = None, log: bool = True) -> dict:
        t0 = time.time()
        retrieved = self.retriever.retrieve(question, top_k=top_k, mode=mode)
        t_ret = time.time() - t0

        context, used = build_context(retrieved, self.s.max_context_chars)
        user_prompt = build_user_prompt(question, context)
        gen = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        answer = gen["text"]
        sources, invalid = parse_citations(answer, used)

        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "abstained": answer.strip().startswith(ABSTAIN),
            "sources": sources,
            "invalid_citations": invalid,
            "retrieval": {
                "mode": (mode or self.s.retrieval_mode),
                "top_k": top_k or self.s.top_k,
                "seconds": round(t_ret, 2),
                "passages": [rc.to_dict() for rc in retrieved],
                "passages_in_prompt": len(used),
                "context_chars": len(context),
            },
            "generation": {
                "provider": self.s.llm_provider,
                "model": self.s.llm_model,
                "temperature": self.s.temperature,
                "seed": self.s.seed,
                "num_ctx": self.s.num_ctx,
                **{k: v for k, v in gen.items() if k != "text"},
            },
            "prompt": {"system": SYSTEM_PROMPT, "user": user_prompt},
        }
        if log:
            self._log(result)
        return result

    def _log(self, result: dict) -> None:
        self.s.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.s.log_dir / "queries.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def render_markdown(result: dict, passage_chars: int = 600) -> str:
    """Human-readable report of one query: answer, sources and the retrieved evidence."""
    r = result["retrieval"]
    g = result["generation"]
    lines = [
        f"### Q: {result['question']}",
        "",
        f"**Answer** ({g['model']}, {g.get('latency_s', '?')} s):",
        "",
        "> " + result["answer"].replace("\n", "\n> "),
        "",
        "**Sources cited:**",
    ]
    if result["sources"]:
        for s in result["sources"]:
            lines.append(f"- [{s['ref']}] *{s['title']}*, p. {s['page']}, section \"{s['section']}\" (`{s['chunk_id']}`)")
    else:
        lines.append("- (none)")
    if result["invalid_citations"]:
        lines.append(f"- WARNING: cited numbers with no matching passage: {result['invalid_citations']}")
    lines += [
        "",
        f"**Retrieved passages** (mode={r['mode']}, top_k={r['top_k']}, {r['passages_in_prompt']} sent to LLM, "
        f"{r['context_chars']} context chars):",
        "",
        "| # | chunk_id | page | section | dense rank (cos) | BM25 rank | fused score |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in r["passages"]:
        c = p["chunk"]
        dense = f"{p['dense_rank']} ({p['dense_score']:.3f})" if p["dense_rank"] else "-"
        bm = f"{p['bm25_rank']}" if p["bm25_rank"] else "-"
        lines.append(f"| {p['rank']} | `{c['chunk_id']}` | {c['page']} | {c['section']} | {dense} | {bm} | {p['score']:.4f} |")
    lines.append("")
    lines.append("<details><summary>Passage text</summary>")
    lines.append("")
    for p in r["passages"]:
        c = p["chunk"]
        text = c["text"] if len(c["text"]) <= passage_chars else c["text"][:passage_chars] + " ..."
        lines.append(f"**[{p['rank']}] `{c['chunk_id']}`**: {text}")
        lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)
