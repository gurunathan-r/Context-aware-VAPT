"""Embedding and persistent ChromaDB storage.

Responsibilities:
    - lazy-load and cache the sentence-transformers encoder (CPU-only),
    - own the PersistentClient collection (cosine space, survives restarts),
    - batch-embed chunk texts and upsert them with sanitized metadata,
    - expose index statistics.

Metadata sanitization: ChromaDB only persists metadata values of type str,
int, float or bool. Any None becomes config.METADATA_NULL_SUBSTITUTE ("unknown")
rather than being dropped silently, so agents always see a well-formed value.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ChromaDB internal type paths vary across versions; only import for typing.
if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

from org_rag_phase1.config import (
    BATCH_SIZE,
    CHROMA_PATH,
    COLLECTION_NAME,
    COSINE_SPACE,
    EMBEDDING_MODEL,
    METADATA_NULL_SUBSTITUTE,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

# Module-level caches — lazy singletons so importing this module stays cheap
# and tests can override paths without loading the model at import time.
_embedder: SentenceTransformer | None = None
_client: chromadb.api.ClientAPI | None = None
_collection: Collection | None = None


def get_embedder() -> SentenceTransformer:
    """Return the shared sentence-transformers embedding model (lazy-loaded).
    The model runs on CPU only and is fully local after the first download,
    keeping indexing/retrieval reproducible and offline-friendly.

    Returns:
        Cached SentenceTransformer instance for config.EMBEDDING_MODEL.
    """
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model %s (CPU)...", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def _get_client() -> chromadb.api.ClientAPI:
    """Return the shared PersistentClient (created on first use)."""
    global _client
    if _client is None:
        persist_dir = PROJECT_ROOT / CHROMA_PATH
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),  # no external calls
        )
    return _client


def get_collection() -> Collection:
    """Return the project collection, creating it if it does not exist.

    The collection is configured with cosine space so stored distances are
    cosine distances in [0, 2] (smaller = more similar).

    Returns:
        The chromadb Collection named config.COLLECTION_NAME.
    """
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": COSINE_SPACE},
        )
        logger.debug("Collection '%s' ready (%d items)", COLLECTION_NAME, _collection.count())
    return _collection


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce a metadata dict into ChromaDB's supported value types.

    ChromaDB persists metadata values only as str, int, float or bool. This
    function converts None to "unknown", stringifies unknown object types, and
    leaves already-supported primitives untouched.

    Args:
        meta: Raw metadata mapping.

    Returns:
        Sanitized copy safe to pass to chromadb.
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = METADATA_NULL_SUBSTITUTE
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:  # lists, dicts, paths, ... -> deterministic string form
            clean[key] = str(value)
    return clean


def index_chunks(chunks: list[dict[str, Any]], reset: bool = False) -> int:
    """Embed chunks in batches and persist them to ChromaDB.

    Uses ``upsert`` semantics, so re-indexing unchanged chunks is idempotent —
    running build_index.py twice neither duplicates nor corrupts state.

    Args:
        chunks: Chunk records as produced by ingest.chunk_document. Each must
            contain ``chunk_id`` and ``chunk_text``.
        reset: If True, delete and recreate the collection before indexing
            (fresh build). If False, upsert into the existing collection.

    Returns:
        Total number of chunks present in the collection after indexing.

    Raises:
        ValueError: If ``chunks`` is empty, or a chunk lacks ``chunk_id`` /
            ``chunk_text``.
    """
    if not chunks:
        raise ValueError("index_chunks called with no chunks — nothing to index.")
    for i, chunk in enumerate(chunks):
        if "chunk_id" not in chunk or "chunk_text" not in chunk:
            raise ValueError(
                f"Chunk at position {i} is missing 'chunk_id'/'chunk_text'; "
                f"got keys: {sorted(chunk)}"
            )

    if reset:
        reset_collection()

    collection = get_collection()
    embedder = get_embedder()

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["chunk_text"] for c in chunks]
    # chunk_text itself is stored as the document; every other field is metadata.
    metadatas = [_sanitize_metadata({k: v for k, v in c.items() if k != "chunk_text"}) for c in chunks]

    logger.info("Embedding %d chunk(s) in batches of %d...", len(ids), BATCH_SIZE)
    embeddings = embedder.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Upserted %d chunk(s); collection now holds %d", len(ids), collection.count())
    return collection.count()


def reset_collection() -> None:
    """Delete and recreate the collection (used by --reset and tests)."""
    global _collection
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:  # chroma raises (ValueError/NotFoundError variants) if absent
        logger.debug("No existing collection '%s' to delete", COLLECTION_NAME)
    _collection = None
    get_collection()


def index_stats() -> dict[str, Any]:
    """Summarize the current collection contents.

    Returns:
        Dict with:
          - ``total_chunks``: number of items in the collection,
          - ``source_type_counts``: items per source_type,
          - ``business_unit_counts``: items per business_unit.
        Empty collections return zero counts with empty dicts.
    """
    collection = get_collection()
    total = collection.count()

    source_type_counts: dict[str, int] = {}
    business_unit_counts: dict[str, int] = {}
    if total > 0:
        included = collection.get(include=["metadatas"]) or {}
        for meta in included.get("metadatas") or []:
            st = str(meta.get("source_type", METADATA_NULL_SUBSTITUTE))
            bu = str(meta.get("business_unit", METADATA_NULL_SUBSTITUTE))
            source_type_counts[st] = source_type_counts.get(st, 0) + 1
            business_unit_counts[bu] = business_unit_counts.get(bu, 0) + 1

    return {
        "total_chunks": total,
        "source_type_counts": source_type_counts,
        "business_unit_counts": business_unit_counts,
    }
