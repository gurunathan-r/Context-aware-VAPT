"""Document loading, sidecar-metadata handling, and semantic chunking.

Phase 1 input contract: every supported document (``.txt`` / ``.md``) in the
raw directory MUST have a sidecar ``<name>.meta.json`` file describing its
organizational context. Missing sidecars are a hard error — a document without
metadata would silently degrade downstream prioritization quality, so we fail
loudly instead of skipping it.

The chunker produces deterministic chunk ids (SHA-256 based) so that repeated
builds over unchanged inputs are idempotent and ChromaDB upserts are stable.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from org_rag_phase1.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RAW_DIR,
    SEPARATORS,
)

logger = logging.getLogger(__name__)

# File extensions considered ingestible. Anything else in data/raw/ is ignored
# (but its sidecar, if present, is still validated by _load_sidecar).
SUPPORTED_EXTENSIONS = {".txt", ".md"}

# Metadata keys every sidecar must provide for every document.
REQUIRED_META_FIELDS = (
    "source_type",
    "business_unit",
    "asset_criticality",
    "compliance_scope",
    "authority_level",
    "source_file",
)


def load_documents(raw_dir: str | Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load all supported raw documents together with their sidecar metadata.

    Args:
        raw_dir: Directory containing ``.txt``/``.md`` documents and their
            ``.meta.json`` sidecars. Defaults to config.RAW_DIR.

    Returns:
        List of ``{"text": str, "meta": dict}`` records, sorted by filename for
        deterministic ordering. Each ``meta`` dict contains at least the keys in
        REQUIRED_META_FIELDS; ``source_file`` is normalized to the file name
        (e.g. ``payment_policy.txt``).

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        ValueError: If a document has no sidecar, or the sidecar is invalid
            JSON / missing required fields. Nothing is silently skipped.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(
            f"Raw documents directory not found: {raw_path.resolve()}"
        )

    doc_files = sorted(
        p
        for p in raw_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not doc_files:
        raise ValueError(
            f"No .txt/.md documents found in {raw_path.resolve()} — nothing to ingest."
        )

    documents: list[dict[str, Any]] = []
    for doc_path in doc_files:
        meta = _load_sidecar(doc_path)
        text = doc_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Document is empty: {doc_path}")
        meta["source_file"] = doc_path.name  # normalize / enforce consistency
        documents.append({"text": text, "meta": meta})
        logger.info("Loaded %s (%d chars)", doc_path.name, len(text))

    logger.info("Loaded %d document(s) from %s", len(documents), raw_path)
    return documents


def _load_sidecar(doc_path: Path) -> dict[str, Any]:
    """Load and validate the ``.meta.json`` sidecar for a document.

    Args:
        doc_path: Path to the primary document.

    Returns:
        The parsed metadata dict.

    Raises:
        ValueError: If the sidecar is missing, unparsable, or missing required
            fields. The error names the exact file so failures are actionable.
    """
    # NOTE: doc.with_suffix(".meta.json") would replace ".txt" instead of
    # appending; the spec'd sidecar name is "<name>.txt.meta.json".
    sidecar = doc_path.parent / f"{doc_path.name}.meta.json"
    if not sidecar.is_file():
        raise ValueError(
            f"Missing sidecar metadata file for {doc_path.name}: expected "
            f"{sidecar.name} in the same directory. Every document must ship "
            f"with organizational metadata (source_type, business_unit, "
            f"asset_criticality, compliance_scope, authority_level)."
        )
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Sidecar {sidecar.name} is not valid JSON: {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"Sidecar {sidecar.name} must contain a JSON object.")

    missing = [f for f in REQUIRED_META_FIELDS if f not in meta or meta[f] is None]
    if missing:
        raise ValueError(
            f"Sidecar {sidecar.name} for {doc_path.name} is missing required "
            f"field(s): {', '.join(missing)}."
        )
    return meta


def chunk_document(text: str, source_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one document into semantically chunked records with full metadata.

    Uses RecursiveCharacterTextSplitter with the project-wide chunking config
    (1000 chars / 200 overlap, paragraph -> sentence -> char separators) so
    chunk boundaries follow document structure rather than hard cuts.

    Args:
        text: Full document text.
        source_meta: Document-level metadata from the sidecar. Must include
            ``source_file`` (see load_documents).

    Returns:
        List of chunk records, each containing ALL source metadata fields plus:
        ``chunk_id`` (deterministic 16-hex-char SHA-256 prefix of
        ``"{source_file}_{chunk_index}"``), ``chunk_text``, ``chunk_index``,
        and ``total_chunks``.

    Raises:
        KeyError: If ``source_meta`` lacks ``source_file``.
        ValueError: If ``text`` is empty.
    """
    if not text or not text.strip():
        raise ValueError("chunk_document called with empty text.")
    if "source_file" not in source_meta:
        raise KeyError(
            "source_meta must include 'source_file' (set by load_documents). "
            f"Got keys: {sorted(source_meta)}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=list(SEPARATORS),
        length_function=len,
    )
    texts = splitter.split_text(text)
    total = len(texts)

    source_file = source_meta["source_file"]
    chunks: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(texts):
        chunk = dict(source_meta)  # all sidecar fields propagate to every chunk
        chunk.update(
            {
                "chunk_id": _chunk_id(source_file, idx),
                "chunk_text": chunk_text,
                "chunk_index": idx,
                "total_chunks": total,
            }
        )
        chunks.append(chunk)

    logger.debug("Chunked %s into %d chunk(s)", source_file, total)
    return chunks


def _chunk_id(source_file: str, chunk_index: int) -> str:
    """Deterministic chunk id: first 16 hex chars of SHA-256 of the nat key.

    Determinism guarantees that re-running the build over unchanged inputs
    produces identical ids, making ChromaDB upserts idempotent.
    """
    digest = hashlib.sha256(f"{source_file}_{chunk_index}".encode("utf-8")).hexdigest()
    return digest[:16]


def ingest_all(raw_dir: str | Path = RAW_DIR) -> list[dict[str, Any]]:
    """Convenience wrapper: load every document and chunk it.

    Args:
        raw_dir: Directory containing raw documents. Defaults to config.RAW_DIR.

    Returns:
        Flattened list of chunk records across all documents (documents are
        processed in sorted filename order for determinism).
    """
    all_chunks: list[dict[str, Any]] = []
    for doc in load_documents(raw_dir):
        all_chunks.extend(chunk_document(doc["text"], doc["meta"]))
    logger.info("Ingested %d chunk(s) total", len(all_chunks))
    return all_chunks
