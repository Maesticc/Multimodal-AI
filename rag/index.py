"""Index building: embed chunks and persist them to ChromaDB.

Artifacts written to `settings.index_dir`:
- chroma/          ChromaDB persistent store (dense vectors + metadata)
- chunks.jsonl     every chunk with its metadata (used for BM25 and inspection)
- manifest.json    settings, model names and corpus statistics used to build the index
"""

from __future__ import annotations

import json
import shutil
import time
from functools import lru_cache

from .config import Settings
from .ingest import Chunk, load_corpus


@lru_cache(maxsize=2)
def get_embedder(model_name: str, device: str):
    # Imported lazily so CLI commands that do not need embeddings start fast.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed_passages(settings: Settings, texts: list[str]) -> list[list[float]]:
    model = get_embedder(settings.embed_model, settings.embed_device)
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    return vecs.tolist()


def embed_query(settings: Settings, query: str) -> list[float]:
    model = get_embedder(settings.embed_model, settings.embed_device)
    vec = model.encode([settings.embed_query_prefix + query], normalize_embeddings=True)
    return vec[0].tolist()


def get_collection(settings: Settings, create: bool = False):
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))
    if create:
        return client.get_or_create_collection(name=settings.collection, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name=settings.collection)


def build_index(settings: Settings) -> dict:
    t0 = time.time()
    chunks, stats = load_corpus(settings)

    # Rebuild from scratch so the index always matches the current settings.
    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)
    settings.index_dir.mkdir(parents=True)

    with open(settings.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    vectors = embed_passages(settings, [c.text for c in chunks])
    col = get_collection(settings, create=True)
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[i : i + batch],
            documents=[c.text for c in part],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source_file": c.source_file,
                    "page": c.page,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                }
                for c in part
            ],
        )

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_seconds": round(time.time() - t0, 1),
        "num_chunks": len(chunks),
        "embedding_dim": len(vectors[0]) if vectors else None,
        "documents": stats,
        "settings": {
            k: v
            for k, v in settings.to_dict().items()
            if k in {"chunk_size", "chunk_overlap", "min_chunk_chars", "include_references", "embed_model", "collection"}
        },
    }
    with open(settings.index_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_chunks(settings: Settings) -> list[Chunk]:
    path = settings.index_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {settings.index_dir}. Run `python -m rag ingest` first.")
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_manifest(settings: Settings) -> dict:
    with open(settings.index_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)
