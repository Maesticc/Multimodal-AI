"""Central configuration.

Every model/provider/retrieval setting is read from environment variables
(optionally loaded from a `.env` file) so nothing is hard-coded in the
pipeline. See `.env.example` for the full list and defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root if present. Real environment variables win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(_env(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    # --- Data / index locations ---
    data_dir: Path = field(default_factory=lambda: _path("RAG_DATA_DIR", "data"))
    sources_file: Path = field(default_factory=lambda: _path("RAG_SOURCES_FILE", "data/sources.json"))
    index_dir: Path = field(default_factory=lambda: _path("RAG_INDEX_DIR", "index"))
    log_dir: Path = field(default_factory=lambda: _path("RAG_LOG_DIR", "logs"))
    collection: str = field(default_factory=lambda: _env("RAG_COLLECTION", "papers"))

    # --- Ingestion ---
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 1000))  # characters
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 200))  # characters
    min_chunk_chars: int = field(default_factory=lambda: _env_int("RAG_MIN_CHUNK_CHARS", 300))
    include_references: bool = field(default_factory=lambda: _env_bool("RAG_INCLUDE_REFERENCES", False))

    # --- Embeddings ---
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    # bge models expect this instruction on the *query* side only.
    embed_query_prefix: str = field(
        default_factory=lambda: _env(
            "RAG_EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
        )
    )
    embed_device: str = field(default_factory=lambda: _env("RAG_EMBED_DEVICE", "cpu"))

    # --- Retrieval ---
    retrieval_mode: str = field(default_factory=lambda: _env("RAG_RETRIEVAL_MODE", "hybrid"))  # dense|bm25|hybrid
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 20))  # per retriever, before fusion
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))

    # --- Generation (LLM) ---
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "llama3.2:3b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    temperature: float = field(default_factory=lambda: _env_float("RAG_TEMPERATURE", 0.0))
    seed: int = field(default_factory=lambda: _env_int("RAG_SEED", 42))
    num_ctx: int = field(default_factory=lambda: _env_int("RAG_NUM_CTX", 4096))  # LLM context window (tokens)
    max_context_chars: int = field(default_factory=lambda: _env_int("RAG_MAX_CONTEXT_CHARS", 6000))
    max_answer_tokens: int = field(default_factory=lambda: _env_int("RAG_MAX_ANSWER_TOKENS", 512))
    request_timeout: int = field(default_factory=lambda: _env_int("RAG_REQUEST_TIMEOUT", 300))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            if not hasattr(s, k):
                raise AttributeError(f"Unknown setting: {k}")
            setattr(s, k, v)
    return s
