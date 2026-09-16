"""Tests for src/index.py: indexing counts, stats, idempotency, sanitization."""

from __future__ import annotations

import pytest

from org_rag_phase1.config import COLLECTION_NAME
from org_rag_phase1.src.index import (
    _sanitize_metadata,
    get_collection,
    index_chunks,
    index_stats,
    reset_collection,
)
from org_rag_phase1.src.ingest import ingest_all


@pytest.fixture()
def sample_chunks():
    """Real chunks from data/raw/ (fast: three small files)."""
    from org_rag_phase1.config import RAW_DIR

    return ingest_all(RAW_DIR)


class TestIndexChunks:
    def test_index_count_matches_chunks(self, index_env, sample_chunks):
        total = index_chunks(sample_chunks, reset=True)
        assert total == len(sample_chunks) > 0
        assert get_collection().count() == len(sample_chunks)

    def test_indexed_documents_match_chunk_text(self, index_env, sample_chunks):
        index_chunks(sample_chunks, reset=True)
        stored = get_collection().get(include=["documents", "metadatas"])
        by_id = dict(zip(stored["ids"], stored["documents"]))
        for chunk in sample_chunks:
            assert by_id[chunk["chunk_id"]] == chunk["chunk_text"]

    def test_metadata_stripped_of_chunk_text(self, index_env, sample_chunks):
        index_chunks(sample_chunks, reset=True)
        stored = get_collection().get(include=["metadatas"])
        for meta in stored["metadatas"]:
            assert "chunk_text" not in meta
            assert meta["source_type"] in {"asset", "policy", "topology"}
            assert meta["asset_criticality"] in {"high", "medium", "low"}

    def test_no_chunk_text_in_metadata(self, index_env, sample_chunks):
        index_chunks(sample_chunks, reset=True)
        stored = get_collection().get(include=["metadatas"])
        assert all("chunk_text" not in m for m in stored["metadatas"])

    def test_empty_input_raises(self, index_env):
        with pytest.raises(ValueError, match="no chunks"):
            index_chunks([], reset=True)

    def test_chunk_missing_fields_raises(self, index_env):
        with pytest.raises(ValueError, match="chunk_id"):
            index_chunks([{"chunk_text": "orphan"}], reset=True)


class TestIdempotency:
    def test_reindex_upserts_without_duplicates(self, index_env, sample_chunks):
        first = index_chunks(sample_chunks, reset=True)
        second = index_chunks(sample_chunks)  # no reset
        assert first == second == len(sample_chunks)
        assert get_collection().count() == len(sample_chunks)

    def test_reset_clears_old_state(self, index_env, sample_chunks):
        index_chunks(sample_chunks, reset=True)
        reset_collection()
        assert get_collection().count() == 0
        index_chunks(sample_chunks)  # rebuild without reset
        assert get_collection().count() == len(sample_chunks)

    def test_collection_name_and_space(self, index_env):
        col = get_collection()
        assert col.name == COLLECTION_NAME
        assert col.metadata.get("hnsw:space") == "cosine"


class TestIndexStats:
    def test_stats_after_indexing(self, index_env, sample_chunks):
        index_chunks(sample_chunks, reset=True)
        stats = index_stats()
        assert stats["total_chunks"] == len(sample_chunks)
        assert set(stats["source_type_counts"]) == {"asset", "policy", "topology"}
        assert sum(stats["source_type_counts"].values()) == len(sample_chunks)
        assert sum(stats["business_unit_counts"].values()) == len(sample_chunks)

    def test_stats_empty_collection(self, index_env):
        reset_collection()
        stats = index_stats()
        assert stats == {
            "total_chunks": 0,
            "source_type_counts": {},
            "business_unit_counts": {},
        }


class TestSanitizeMetadata:
    def test_none_becomes_unknown(self):
        assert _sanitize_metadata({"a": None})["a"] == "unknown"

    def test_primitives_pass_through(self):
        meta = {"s": "x", "i": 1, "f": 1.5, "b": True}
        assert _sanitize_metadata(meta) == meta

    def test_unsupported_types_stringified(self):
        out = _sanitize_metadata({"tags": ["a", "b"], "path": object()})
        assert out["tags"] == "['a', 'b']"
        assert isinstance(out["path"], str)
