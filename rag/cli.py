"""Command-line interface.

    python -m rag ingest                      # build the index from data/*.pdf
    python -m rag ask "question" [--show-context] [--mode hybrid|dense|bm25] [--top-k 5] [--json]
    python -m rag retrieve "question"         # retrieval only, no LLM
    python -m rag chat                        # interactive loop
    python -m rag examples                    # run examples/questions.json, save outputs
    python -m rag eval                        # retrieval evaluation (dense vs BM25 vs hybrid)
    python -m rag info                        # show configuration, index and model details
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .config import get_settings


def _settings(args):
    return get_settings(
        top_k=getattr(args, "top_k", None),
        retrieval_mode=getattr(args, "mode", None),
        llm_model=getattr(args, "model", None),
    )


def cmd_ingest(args) -> None:
    from .index import build_index

    s = get_settings(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    m = build_index(s)
    print(f"Indexed {m['num_chunks']} chunks (dim={m['embedding_dim']}) in {m['build_seconds']} s -> {s.index_dir}")
    for d in m["documents"]:
        print(f"  {d['doc_id']:<18} pages={d['pages']:<3} chunks={d['chunks']:<4} chars={d['chars']}")


def _print_passages(passages) -> None:
    for p in passages:
        c = p.chunk
        dense = f"{p.dense_rank}/{p.dense_score:.3f}" if p.dense_rank else "-"
        print(f"\n[{p.rank}] {c.chunk_id}  page={c.page}  section={c.section!r}")
        print(f"    score={p.score:.4f}  dense(rank/cos)={dense}  bm25_rank={p.bm25_rank or '-'}")
        print(textwrap.indent(textwrap.fill(c.text[:500] + (" ..." if len(c.text) > 500 else ""), 100), "    "))


def cmd_retrieve(args) -> None:
    from .retrieve import Retriever

    s = _settings(args)
    res = Retriever(s).retrieve(args.question)
    if args.json:
        print(json.dumps([r.to_dict() for r in res], indent=2, ensure_ascii=False))
    else:
        print(f"mode={s.retrieval_mode} top_k={s.top_k}")
        _print_passages(res)


def _print_result(res: dict, show_context: bool) -> None:
    print("\n" + "=" * 100)
    print("ANSWER:\n")
    print(textwrap.fill(res["answer"], 100, replace_whitespace=False))
    print("\nSOURCES:")
    for s in res["sources"]:
        print(f"  [{s['ref']}] {s['title']} | p. {s['page']} | {s['section']} | {s['chunk_id']}")
    if not res["sources"]:
        print("  (no citations in answer)")
    if res["invalid_citations"]:
        print(f"  WARNING: answer cites non-existent passages {res['invalid_citations']}")
    g, r = res["generation"], res["retrieval"]
    print(
        f"\n[{g['model']} | retrieval {r['seconds']} s | generation {g['latency_s']} s | "
        f"prompt tokens {g['prompt_tokens']} | {r['passages_in_prompt']}/{len(r['passages'])} passages in prompt]"
    )
    if show_context:
        print("\nRETRIEVED PASSAGES:")
        for p in r["passages"]:
            c = p["chunk"]
            print(f"\n[{p['rank']}] {c['chunk_id']} page={c['page']} section={c['section']!r} score={p['score']:.4f}")
            print(textwrap.indent(textwrap.fill(c["text"], 100), "    "))
    print("=" * 100)


def cmd_ask(args) -> None:
    from .pipeline import RAGPipeline

    res = RAGPipeline(_settings(args)).ask(args.question)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        _print_result(res, args.show_context)


def cmd_chat(args) -> None:
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(_settings(args))
    print("Ask a question about the papers (empty line or 'exit' to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        _print_result(pipe.ask(q), args.show_context)


def cmd_examples(args) -> None:
    from .examples import run_examples

    run_examples(_settings(args))


def cmd_eval(args) -> None:
    from .evaluate import run_eval

    run_eval(get_settings(), ks=(1, 3, 5))


def cmd_info(args) -> None:
    from .generate import get_llm
    from .index import load_manifest

    s = get_settings()
    print("Settings:")
    print(json.dumps(s.to_dict(), indent=2))
    try:
        print("\nIndex manifest:")
        print(json.dumps(load_manifest(s), indent=2))
    except FileNotFoundError:
        print("  (no index yet - run `python -m rag ingest`)")
    print("\nLLM:")
    print(json.dumps(get_llm(s).model_info(), indent=2))


def main(argv=None) -> None:
    # Windows consoles default to cp1252; papers contain unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(prog="python -m rag", description="RAG over research papers")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="Build the vector index")
    sp.add_argument("--chunk-size", type=int)
    sp.add_argument("--chunk-overlap", type=int)
    sp.set_defaults(func=cmd_ingest)

    def common(sp):
        sp.add_argument("--mode", choices=["dense", "bm25", "hybrid"])
        sp.add_argument("--top-k", type=int)
        sp.add_argument("--model", help="Override LLM model name")

    sp = sub.add_parser("ask", help="Answer a question")
    sp.add_argument("question")
    sp.add_argument("--show-context", action="store_true")
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("retrieve", help="Retrieval only")
    sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_retrieve)

    sp = sub.add_parser("chat", help="Interactive mode")
    sp.add_argument("--show-context", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("examples", help="Run example questions and save outputs")
    common(sp)
    sp.set_defaults(func=cmd_examples)

    sp = sub.add_parser("eval", help="Retrieval evaluation")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("info", help="Show config / index / model")
    sp.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
