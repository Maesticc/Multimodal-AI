"""UI for the RAG pipeline.

This file only renders results. Retrieval, prompting, generation and citation
parsing all happen in `rag.pipeline.RAGPipeline.ask()`, the same code path the
CLI uses.
"""

from __future__ import annotations

import html
import json
import re

import streamlit as st

from rag.config import PROJECT_ROOT, get_settings
from rag.generate import get_llm
from rag.index import load_manifest
from rag.pipeline import RAGPipeline

SAVED_ANSWERS = PROJECT_ROOT / "examples" / "outputs" / "answers.jsonl"

st.set_page_config(page_title="PaperRAG", page_icon="✦", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root { --accent: linear-gradient(90deg,#7c5cff,#22d3ee); --card:#141b33; --line:#232c4d; --muted:#8a93b8; }
.stApp { background: linear-gradient(135deg,#0b1020 0%,#111a33 100%); }
[data-testid="stSidebar"] { background:#10162c; border-right:1px solid var(--line); }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
h1,h2,h3 { letter-spacing:-0.01em; }

.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:1.2rem; }
.brand-logo { width:36px; height:36px; border-radius:10px; background:var(--accent);
  display:flex; align-items:center; justify-content:center; font-weight:800; color:#0b1020; }
.brand-title { font-weight:700; font-size:1rem; line-height:1.1; }
.brand-sub { color:var(--muted); font-size:.75rem; }

.hero { background:linear-gradient(135deg,#1b1f4a,#0f2a3a); border:1px solid #2a3466;
  border-radius:18px; padding:1.4rem 1.6rem 1rem; margin-bottom:1rem; position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-80px; width:240px; height:240px;
  border-radius:50%; background:#7c5cff; opacity:.12; }
.hero h1 { font-size:1.7rem; font-weight:800; margin:0; color:#fff; }
.hero p { color:#a9b2d6; margin:.3rem 0 0; font-size:.9rem; }

.card { background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:1.1rem 1.25rem; box-shadow:0 8px 24px rgba(0,0,0,.25); height:100%; }
.label { font-size:.7rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase; margin-bottom:.6rem;
  display:flex; justify-content:space-between; align-items:center; }
.answer { font-size:.97rem; line-height:1.6; color:#e8ecff; white-space:pre-wrap; }
.cite { display:inline-block; min-width:1.4rem; padding:0 .4rem; border-radius:999px; background:#2a2366;
  color:#b9a8ff; font-size:.72rem; font-weight:700; text-align:center; margin:0 .1rem; }
.cite.bad { background:#3a1d2a; color:#ff8fa3; }
.badge { padding:.15rem .6rem; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:0; text-transform:none; }
.badge.ok { background:#163b2e; color:#34d399; }
.badge.abstain { background:#3b3216; color:#fbbf24; }
.badge.warn { background:#3a1d2a; color:#ff8fa3; }
.meta { color:#6b7499; font-size:.78rem; margin-top:.9rem; padding-top:.7rem; border-top:1px solid var(--line); }
.meta span { margin-right:1.1rem; }

.src { background:#0f152b; border:1px solid #2a2366; border-radius:12px; padding:.7rem .8rem; margin-bottom:.6rem; }
.src-title { font-weight:700; font-size:.88rem; color:#e8ecff; }
.src-line { color:#a9b2d6; font-size:.8rem; margin-top:.15rem; }
.src-id { color:#6b7499; font-size:.72rem; margin-top:.2rem; font-family:ui-monospace,Consolas,monospace; }
.hint { color:#6b7499; font-size:.78rem; }

.passage { background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:.85rem 1rem .85rem 1.15rem; margin-bottom:.6rem; position:relative; display:flex; gap:1rem; }
.passage.cited { border-color:#7c5cff; background:#161f40; }
.passage.cited:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  border-radius:14px 0 0 14px; background:var(--accent); }
.p-main { flex:1; min-width:0; }
.p-head { font-weight:700; font-size:.9rem; color:#e8ecff; }
.p-sec { color:#6b7499; font-size:.75rem; margin:.15rem 0 .35rem; }
.p-text { color:#a9b2d6; font-size:.82rem; line-height:1.5; }
.p-scores { width:190px; flex-shrink:0; font-size:.7rem; color:var(--muted); }
.bar-row { display:flex; align-items:center; gap:.45rem; margin:.28rem 0; }
.bar-row .k { width:38px; }
.bar { flex:1; height:6px; border-radius:3px; background:#232c4d; overflow:hidden; }
.bar > i { display:block; height:100%; border-radius:3px; }
.bar-row .v { width:44px; text-align:right; color:#c9cff0; font-family:ui-monospace,Consolas,monospace; }

.sys { background:#0f152b; border:1px solid var(--line); border-radius:12px; padding:.75rem .85rem; font-size:.8rem; }
.sys-row { display:flex; justify-content:space-between; margin:.25rem 0; color:var(--muted); }
.sys-row b { color:#e8ecff; font-weight:500; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#7c5cff,#22d3ee); color:#0b1020;
  border:0; font-weight:700; border-radius:999px; }
.stTextInput input { border-radius:999px !important; padding:.65rem 1.1rem !important; background:#0b1020 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources: load embeddings, BM25 and Chroma once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index and models…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())


@st.cache_data(ttl=30, show_spinner=False)
def get_model_info() -> dict:
    return get_llm(get_settings()).model_info()


@st.cache_data(show_spinner=False)
def get_saved_examples() -> list[dict]:
    if not SAVED_ANSWERS.exists():
        return []
    with open(SAVED_ANSWERS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
_CITE = re.compile(r"\[(\d+(?:\s*[,;-]\s*\d+)*)\]")


def render_answer_html(answer: str, n_passages: int) -> str:
    """Escape the answer, then turn [n] markers into chips (red if they match no passage)."""

    def chip(m: re.Match) -> str:
        nums = [p.strip() for p in re.split(r"[,;]", m.group(1)) if p.strip()]
        out = []
        for n in nums:
            ok = "-" not in n and 1 <= int(n) <= n_passages
            out.append(f'<span class="cite{"" if ok else " bad"}">{n}</span>')
        return "".join(out)

    return _CITE.sub(chip, html.escape(answer))


def status_badge(res: dict) -> str:
    if res["abstained"]:
        return '<span class="badge abstain">● abstained</span>'
    if res["invalid_citations"]:
        return '<span class="badge warn">● check citations</span>'
    if not res["sources"]:
        return '<span class="badge warn">● no citations</span>'
    return '<span class="badge ok">● cited</span>'


def bar(label: str, value: float, display: str, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f'<div class="bar-row"><span class="k">{label}</span>'
        f'<span class="bar"><i style="width:{pct:.0f}%;background:{color}"></i></span>'
        f'<span class="v">{display}</span></div>'
    )


def render_passage(p: dict, cited: bool, bm25_max: float, rrf_max: float) -> str:
    c = p["chunk"]
    text = c["text"] if len(c["text"]) <= 420 else c["text"][:420] + " …"
    dense = (
        bar("dense", p["dense_score"], f'#{p["dense_rank"]}', "#7c5cff")
        if p.get("dense_rank")
        else bar("dense", 0, "–", "#7c5cff")
    )
    bm25 = (
        bar("bm25", (p["bm25_score"] or 0) / bm25_max if bm25_max else 0, f'#{p["bm25_rank"]}', "#22d3ee")
        if p.get("bm25_rank")
        else bar("bm25", 0, "–", "#22d3ee")
    )
    fused = bar("final", p["score"] / rrf_max if rrf_max else 0, f'{p["score"]:.3f}', "#e8ecff")
    tag = ' <span class="cite">★ cited</span>' if cited else ""
    title = c["title"].split(":")[0]
    return (
        f'<div class="passage{" cited" if cited else ""}"><div class="p-main">'
        f'<div class="p-head">#{p["rank"]} &nbsp;{html.escape(title)} · p.{c["page"]}{tag}</div>'
        f'<div class="p-sec">{html.escape(c["section"])} · <code>{c["chunk_id"]}</code></div>'
        f'<div class="p-text">{html.escape(text)}</div></div>'
        f'<div class="p-scores">{dense}{bm25}{fused}</div></div>'
    )


def show_result(res: dict) -> None:
    r, g = res["retrieval"], res["generation"]
    n_in_prompt = r["passages_in_prompt"]

    col_a, col_s = st.columns([1.85, 1], gap="medium")
    with col_a:
        st.markdown(
            f'<div class="card"><div class="label"><span>Answer</span>{status_badge(res)}</div>'
            f'<div class="answer">{render_answer_html(res["answer"], n_in_prompt)}</div>'
            f'<div class="meta"><span>⏱ {r["seconds"]} s retrieval</span>'
            f'<span>⚡ {g.get("latency_s", "?")} s generation</span>'
            f'<span>▦ {g.get("prompt_tokens") or "?"} prompt tokens</span>'
            f'<span>◉ {n_in_prompt}/{len(r["passages"])} passages in prompt</span></div></div>',
            unsafe_allow_html=True,
        )
    with col_s:
        if res["sources"]:
            items = "".join(
                f'<div class="src"><span class="cite">{s["ref"]}</span> '
                f'<span class="src-title">{html.escape(s["title"].split(":")[0])}</span>'
                f'<div class="src-line">Page {s["page"]}</div>'
                f'<div class="src-line">{html.escape(s["section"])}</div>'
                f'<div class="src-id">{s["chunk_id"]}</div></div>'
                for s in res["sources"]
            )
        else:
            items = '<div class="hint">The answer cites no passage.</div>'
        bad = (
            f'<div class="hint" style="color:#ff8fa3">Cited numbers with no matching passage: '
            f'{res["invalid_citations"]}</div>'
            if res["invalid_citations"]
            else ""
        )
        st.markdown(
            f'<div class="card"><div class="label"><span>Sources</span></div>{items}{bad}'
            f'<div class="hint">Only passages that were in the prompt can appear here.</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        '<div class="label"><span>Retrieved passages</span><span>dense rank · BM25 rank · final score</span></div>',
        unsafe_allow_html=True,
    )
    cited_ids = {s["chunk_id"] for s in res["sources"]}
    passages = r["passages"]
    bm25_max = max((p["bm25_score"] or 0 for p in passages), default=0) or 1
    rrf_max = max((p["score"] for p in passages), default=0) or 1
    cards = [render_passage(p, p["chunk"]["chunk_id"] in cited_ids, bm25_max, rrf_max) for p in passages]

    st.markdown("".join(cards[:2]), unsafe_allow_html=True)
    if len(cards) > 2:
        with st.expander(f"Show passages #3–#{len(cards)}"):
            st.markdown("".join(cards[2:]), unsafe_allow_html=True)
    with st.expander("{ }  Inspect the prompt sent to the LLM"):
        st.caption("System prompt")
        st.code(res["prompt"]["system"], language="text", wrap_lines=True)
        st.caption("User prompt (retrieved passages + question)")
        st.code(res["prompt"]["user"], language="text", wrap_lines=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
settings = get_settings()
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">R</div><div>'
        '<div class="brand-title">PaperRAG</div><div class="brand-sub">local · cited answers</div></div></div>',
        unsafe_allow_html=True,
    )
    view = st.radio("View", ["✦  Ask", "▤  Saved examples"], label_visibility="collapsed")

    st.markdown('<div class="label" style="margin-top:1.2rem">Retrieval</div>', unsafe_allow_html=True)
    mode = st.segmented_control(
        "Mode", ["hybrid", "dense", "bm25"], default=settings.retrieval_mode, label_visibility="collapsed"
    ) or settings.retrieval_mode
    top_k = st.slider("Passages to LLM", 1, 10, settings.top_k)

    st.markdown('<div class="label" style="margin-top:1.2rem">System</div>', unsafe_allow_html=True)
    info = get_model_info()
    online = "error" not in info
    try:
        n_chunks = load_manifest(settings)["num_chunks"]
        idx = f"4 papers · {n_chunks}"
    except FileNotFoundError:
        idx = "not built"
    st.markdown(
        f'<div class="sys"><div><span class="dot" style="background:{"#34d399" if online else "#ff8fa3"}"></span>'
        f'Ollama {"connected" if online else "offline"}</div>'
        f'<div class="sys-row">LLM <b>{settings.llm_model}</b></div>'
        f'<div class="sys-row">Embed <b>{settings.embed_model.split("/")[-1].replace("-en-v1.5", "")}</b></div>'
        f'<div class="sys-row">Index <b>{idx}</b></div>'
        f'<div class="sys-row" style="font-size:.72rem">{info.get("quantization") or "–"} · '
        f"temp {settings.temperature} · seed {settings.seed}</div></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>Ask the papers</h1>'
    "<p>MMVP · HallusionBench · RT-2 · Diffusion Policy — every claim is cited to a page.</p></div>",
    unsafe_allow_html=True,
)

if view.endswith("Ask"):
    with st.form("ask", border=False):
        c1, c2 = st.columns([6, 1])
        question = c1.text_input(
            "Question",
            placeholder="e.g. What limitations of Diffusion Policy do the authors acknowledge?",
            label_visibility="collapsed",
        )
        submitted = c2.form_submit_button("Ask →", type="primary", use_container_width=True)

    if submitted and question.strip():
        if not online:
            st.error(f"Cannot reach Ollama at {settings.ollama_base_url}. Start Ollama and try again.")
        else:
            try:
                with st.spinner("Retrieving passages and generating an answer…"):
                    st.session_state["result"] = get_pipeline().ask(question.strip(), top_k=top_k, mode=mode)
            except FileNotFoundError as e:
                st.error(str(e))
            except RuntimeError as e:
                st.error(str(e))

    if "result" in st.session_state:
        show_result(st.session_state["result"])
    else:
        st.markdown(
            '<div class="hint">Ask a question, or open <b>Saved examples</b> in the sidebar to browse the 17 '
            "recorded runs (works without Ollama).</div>",
            unsafe_allow_html=True,
        )
else:
    examples = get_saved_examples()
    if not examples:
        st.info("No saved examples found. Generate them with the `examples` command.")
    else:
        labels = [f'{e["example_id"]} · {e["question"]}' for e in examples]
        choice = st.selectbox("Example", labels, label_visibility="collapsed")
        ex = examples[labels.index(choice)]
        d = ex.get("diagnosis", {})
        st.caption(f'Type: {ex.get("example_type", "–")} · automatic verdict: {d.get("verdict", "–")} · '
                   f'see examples/outputs/manual_review.md for the manual check')
        show_result(ex)
