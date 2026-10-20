"""Document ingestion: PDF -> cleaned page text -> section-aware chunks.

Pipeline per PDF
1. Extract text page by page with pypdf (keeps 1-based page numbers).
2. Walk the lines of each page, detecting section headings with a regex
   (e.g. "2.1. Finding CLIP-blind Pairs", "IV. EVALUATION", "Abstract").
   The current section carries over across pages.
3. Repair common extraction artefacts (hyphenated line breaks, ligatures,
   whitespace) and split into sentences.
4. Greedily pack sentences into chunks of ~`chunk_size` characters with
   ~`chunk_overlap` characters of sentence-level overlap. Chunks never cross
   a page boundary, so every chunk has exactly one page number to cite.
   A section change starts a new chunk once the current one is at least
   `min_chunk_chars` long.
5. By default, text after a "References"/"Bibliography" heading is skipped
   until an appendix-style heading appears, because citation lists add noise
   to retrieval without answering content questions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from .config import Settings

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str  # "<doc_id>:p<page>:c<n>" - unique and human readable
    doc_id: str
    title: str
    source_file: str
    page: int  # 1-based PDF page index
    section: str
    chunk_index: int  # running index within the document
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Source registry (titles / provenance)
# ---------------------------------------------------------------------------


def load_sources(settings: Settings) -> dict[str, dict]:
    """Map PDF filename -> provenance metadata from data/sources.json."""
    if not settings.sources_file.exists():
        return {}
    with open(settings.sources_file, encoding="utf-8") as f:
        entries = json.load(f)["documents"]
    return {e["file"]: e for e in entries}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


# ---------------------------------------------------------------------------
# Text cleaning and heading detection
# ---------------------------------------------------------------------------

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}

# Numbered headings: "1 Introduction", "2.1. Finding CLIP-blind Pairs",
# roman numerals "IV. EVALUATION", appendix letters "A. Details".
_NUMBERED_HEADING = re.compile(
    r"^(?P<num>(\d{1,2}(\.\d{1,2}){0,2}\.?)|([IVX]{1,5}\.)|([A-H](\.\d{1,2})*\.))\s+"
    r"(?P<title>[A-Z][A-Za-z0-9\-:,&/'()\s]{2,70})$"
)
_NAMED_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "conclusion",
    "conclusions",
    "discussion",
    "limitations",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
    "appendix",
}
_REFERENCE_HEADINGS = {"references", "bibliography"}
_STOPWORDS = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "via", "from", "by", "vs", "or"}


def normalize_text(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    return text


def detect_heading(line: str) -> str | None:
    """Return a normalised heading string if `line` looks like a section heading."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    low = s.lower().rstrip(".:")
    if low in _NAMED_HEADINGS:
        return s.rstrip(".:").title() if s.isupper() else s.rstrip(".:")
    m = _NUMBERED_HEADING.match(s)
    if not m:
        return None
    title = m.group("title").strip()
    words = title.split()
    # Reject things that look like sentences, list items, affiliations or figure fragments.
    if title.endswith(".") or " ," in title or len(words) > 10:
        return None
    if any(w.isdigit() for w in words) or (len(words) == 1 and len(title) < 4):
        return None
    content_words = [w for w in words if w.lower() not in _STOPWORDS]
    lowercase = [w for w in content_words if w[0].islower()]
    if content_words and len(lowercase) / len(content_words) > 0.4:
        return None
    # Small-caps headings extract as "I NTRODUCTION" / "K EY DESIGN"; re-join them.
    if title.replace(" ", "").isupper():
        title = re.sub(r"\b([A-Z]) (?=[A-Z]{2,})", r"\1", title).title()
    return f"{m.group('num').rstrip('.')} {title}"


def clean_block(text: str) -> str:
    """Join lines of a block into flowing prose and repair extraction artefacts."""
    # Re-join words hyphenated across line breaks: "rea-\nsoning" -> "reasoning".
    # Heuristic: a lowercase continuation is treated as a soft hyphen, so genuine
    # compounds broken at the hyphen ("vision-\nonly") also lose it ("visiononly").
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(“\"])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        # Hard-wrap pathological "sentences" (tables, equations, missing spaces).
        while len(p) > 600:
            cut = p.rfind(" ", 0, 600)
            cut = cut if cut > 200 else 600
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _pack(
    sentences: list[tuple[str, str]],  # (sentence, section)
    chunk_size: int,
    overlap: int,
    min_chars: int,
) -> list[tuple[str, str]]:
    """Greedy sentence packing with sentence-level overlap. Returns (text, section)."""
    chunks: list[tuple[str, str]] = []
    cur: list[tuple[str, str]] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append((" ".join(s for s, _ in cur), cur[0][1]))
        # Carry over trailing sentences (<= overlap chars) into the next chunk.
        carry: list[tuple[str, str]] = []
        n = 0
        for s, sec in reversed(cur):
            if n + len(s) > overlap:
                break
            carry.insert(0, (s, sec))
            n += len(s) + 1
        cur, cur_len = carry, n

    for sent, sec in sentences:
        section_changed = bool(cur) and sec != cur[-1][1]
        if section_changed and cur_len >= min_chars:
            flush()
            # Do not carry overlap from the previous section into a new one.
            cur, cur_len = [], 0
        if cur and cur_len + len(sent) + 1 > chunk_size:
            flush()
        cur.append((sent, sec))
        cur_len += len(sent) + 1

    if cur:
        # Avoid emitting a chunk that is only overlap from the previous chunk.
        text = " ".join(s for s, _ in cur)
        if not chunks or not chunks[-1][0].endswith(text):
            chunks.append((text, cur[0][1]))
    return chunks


def chunk_pdf(path: Path, settings: Settings, meta: dict | None = None) -> list[Chunk]:
    meta = meta or {}
    reader = PdfReader(str(path))
    doc_id = meta.get("doc_id") or _slug(path.stem)
    title = meta.get("title") or (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)

    section = "Front matter"
    in_references = False
    chunks: list[Chunk] = []
    idx = 0

    for page_no, page in enumerate(reader.pages, start=1):
        raw = normalize_text(page.extract_text() or "")
        blocks: list[tuple[str, str]] = []  # (section, raw text)
        buf: list[str] = []

        def end_block() -> None:
            if buf and not in_references:
                blocks.append((section, "\n".join(buf)))
            buf.clear()

        for line in raw.splitlines():
            heading = detect_heading(line)
            if heading:
                end_block()
                low = heading.lower()
                if low in _REFERENCE_HEADINGS:
                    in_references = not settings.include_references
                elif in_references and re.match(r"^([A-H](\.\d+)*|appendix)\b", heading, re.I):
                    in_references = False  # appendix after the bibliography
                section = heading
                buf.append(line)  # keep the heading text inside the chunk
            else:
                buf.append(line)
        end_block()

        sentences: list[tuple[str, str]] = []
        for sec, text in blocks:
            for s in split_sentences(clean_block(text)):
                sentences.append((s, sec))

        for text, sec in _pack(sentences, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars):
            if len(text) < 50:  # stray figure labels, page numbers
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:p{page_no}:c{idx}",
                    doc_id=doc_id,
                    title=title,
                    source_file=path.name,
                    page=page_no,
                    section=sec,
                    chunk_index=idx,
                    text=text,
                )
            )
            idx += 1
    return chunks


def load_corpus(settings: Settings) -> tuple[list[Chunk], list[dict]]:
    """Chunk every PDF in the data directory. Returns chunks and per-document stats."""
    sources = load_sources(settings)
    pdfs = sorted(settings.data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {settings.data_dir}")
    all_chunks: list[Chunk] = []
    stats: list[dict] = []
    for pdf in pdfs:
        chunks = chunk_pdf(pdf, settings, sources.get(pdf.name))
        all_chunks.extend(chunks)
        n_pages = len(PdfReader(str(pdf)).pages)
        stats.append(
            {
                "file": pdf.name,
                "doc_id": chunks[0].doc_id if chunks else None,
                "title": chunks[0].title if chunks else None,
                "pages": n_pages,
                "chunks": len(chunks),
                "chars": sum(len(c.text) for c in chunks),
                "size_kb": round(pdf.stat().st_size / 1024),
            }
        )
    return all_chunks, stats
