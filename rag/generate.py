"""Answer generation: build a grounded prompt from retrieved chunks and call the LLM.

Prompt design
- Retrieved chunks are numbered [1]..[k] in retrieval order. Each block header
  carries the document title, page and section so the model (and a reviewer
  reading the logged prompt) can see where the evidence came from.
- The system prompt asks the model to answer *only* from the numbered context,
  cite block numbers inline, and reply with a fixed abstention sentence when the
  context is insufficient.
- Context is capped at `max_context_chars` characters (~4 chars/token, so the
  default 6000 chars ~= 1500 tokens). Whole blocks are dropped from the tail
  (lowest-ranked first) rather than truncating mid-passage; the top block is
  always kept, truncated if necessary. With num_ctx=4096 this leaves room for
  the system prompt, the question and a 512-token answer.

Citations in the answer are parsed back to the exact chunks that were shown to
the model, so every reference in the output points at evidence that was
actually retrieved.
"""

from __future__ import annotations

import re
import time

import requests

from .config import Settings
from .retrieve import RetrievedChunk

ABSTAIN = "I cannot answer this from the provided documents."

SYSTEM_PROMPT = f"""You are a research assistant answering questions about a small collection of research papers.

Rules:
1. Use ONLY the information in the numbered context passages. Do not use outside knowledge.
2. After each claim, cite the passage(s) that support it with bracketed numbers, e.g. [1] or [2][3].
3. If the passages do not contain the information needed to answer, reply exactly: "{ABSTAIN}"
   You may then add one sentence saying what related information the passages do contain.
4. Be concise: answer in at most 6 sentences. Quote numbers exactly as they appear in the passages."""


def format_block(n: int, rc: RetrievedChunk, text: str | None = None) -> str:
    c = rc.chunk
    return f"[{n}] (source: {c.title} | page {c.page} | section: {c.section})\n{text if text is not None else c.text}"


def build_context(retrieved: list[RetrievedChunk], max_chars: int) -> tuple[str, list[RetrievedChunk]]:
    """Return the context string and the list of chunks actually included (in citation order)."""
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    for rc in retrieved:
        block = format_block(len(used) + 1, rc)
        if total + len(block) > max_chars:
            if not used:  # always keep the best passage, truncated
                block = format_block(1, rc, rc.chunk.text[: max(200, max_chars - 200)] + " ...")
            else:
                break
        blocks.append(block)
        used.append(rc)
        total += len(block) + 2
    return "\n\n".join(blocks), used


def build_user_prompt(question: str, context: str) -> str:
    return f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer (with [n] citations):"


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.s = settings

    def model_info(self) -> dict:
        """Exact model identity reported by the Ollama server (digest, quantization)."""
        try:
            r = requests.post(f"{self.s.ollama_base_url}/api/show", json={"model": self.s.llm_model}, timeout=30)
            r.raise_for_status()
            j = r.json()
            det = j.get("details", {})
            tags = requests.get(f"{self.s.ollama_base_url}/api/tags", timeout=30).json().get("models", [])
            digest = next((m.get("digest") for m in tags if m.get("name") == self.s.llm_model), None)
            ver = requests.get(f"{self.s.ollama_base_url}/api/version", timeout=30).json().get("version")
            return {
                "provider": "ollama",
                "ollama_version": ver,
                "model": self.s.llm_model,
                "digest": digest,
                "family": det.get("family"),
                "parameter_size": det.get("parameter_size"),
                "quantization": det.get("quantization_level"),
            }
        except requests.RequestException as e:
            return {"provider": "ollama", "model": self.s.llm_model, "error": str(e)}

    def chat(self, system: str, user: str) -> dict:
        payload = {
            "model": self.s.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": self.s.temperature,
                "seed": self.s.seed,
                "num_ctx": self.s.num_ctx,
                "num_predict": self.s.max_answer_tokens,
            },
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.s.ollama_base_url}/api/chat", json=payload, timeout=self.s.request_timeout)
            r.raise_for_status()
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.s.ollama_base_url}. Is it running? (`ollama serve`)"
            ) from e
        j = r.json()
        return {
            "text": j["message"]["content"].strip(),
            "latency_s": round(time.time() - t0, 2),
            "prompt_tokens": j.get("prompt_eval_count"),
            "completion_tokens": j.get("eval_count"),
        }


def get_llm(settings: Settings):
    if settings.llm_provider.lower() == "ollama":
        return OllamaLLM(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


_CITE = re.compile(r"\[(\d+(?:\s*[,;-]\s*\d+)*)\]")


def parse_citations(answer: str, used: list[RetrievedChunk]) -> tuple[list[dict], list[int]]:
    """Map [n] markers in the answer back to the chunks shown to the model.

    Returns (sources, invalid) where `invalid` are cited numbers with no matching block.
    """
    nums: list[int] = []
    for m in _CITE.finditer(answer):
        for part in re.split(r"[,;]", m.group(1)):
            part = part.strip()
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                nums.extend(range(a, b + 1))
            elif part:
                nums.append(int(part))
    seen: list[int] = []
    for n in nums:
        if n not in seen:
            seen.append(n)
    sources, invalid = [], []
    for n in seen:
        if 1 <= n <= len(used):
            c = used[n - 1].chunk
            sources.append(
                {"ref": n, "title": c.title, "file": c.source_file, "page": c.page, "section": c.section, "chunk_id": c.chunk_id}
            )
        else:
            invalid.append(n)
    return sources, invalid
