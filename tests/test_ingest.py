"""Tests for src/ingest.py: loading, sidecar validation, chunking determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from org_rag_phase1.src.ingest import (
    _chunk_id,
    chunk_document,
    ingest_all,
    load_documents,
)

REQUIRED_META = {
    "source_type": "policy",
    "business_unit": "finance",
    "asset_criticality": "high",
    "compliance_scope": "PCI-DSS",
    "authority_level": "mandatory",
}


def _make_doc(dir_path: Path, name: str, text: str, meta_overrides: dict | None = None,
              with_sidecar: bool = True) -> Path:
    """Helper: write a document (and optional sidecar) into dir_path."""
    doc = dir_path / name
    doc.write_text(text, encoding="utf-8")
    if with_sidecar:
        meta = dict(REQUIRED_META)
        meta.update(meta_overrides or {})
        meta["source_file"] = name
        (dir_path / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return doc


class TestLoadDocuments:
    def test_loads_txt_and_md_with_normalized_source_file(self, tmp_path):
        _make_doc(tmp_path, "a.txt", "hello world", {"source_type": "asset"})
        _make_doc(tmp_path, "b.md", "# markdown doc", {"source_type": "policy"})
        docs = load_documents(tmp_path)
        assert [d["meta"]["source_file"] for d in docs] == ["a.txt", "b.md"]
        assert docs[0]["text"] == "hello world"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nope")

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No .txt/.md documents"):
            load_documents(tmp_path)

    def test_missing_sidecar_raises_not_skipped(self, tmp_path):
        _make_doc(tmp_path, "orphan.txt", "no metadata here", with_sidecar=False)
        with pytest.raises(ValueError, match="Missing sidecar"):
            load_documents(tmp_path)

    def test_sidecar_missing_required_field_raises(self, tmp_path):
        _make_doc(tmp_path, "bad.txt", "content")
        sidecar = tmp_path / "bad.txt.meta.json"
        meta = json.loads(sidecar.read_text())
        del meta["compliance_scope"]
        sidecar.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="compliance_scope"):
            load_documents(tmp_path)

    def test_invalid_json_sidecar_raises(self, tmp_path):
        (tmp_path / "broken.txt").write_text("content")
        (tmp_path / "broken.txt.meta.json").write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_documents(tmp_path)

    def test_empty_document_raises(self, tmp_path):
        _make_doc(tmp_path, "empty.txt", "   \n  ")
        with pytest.raises(ValueError, match="empty"):
            load_documents(tmp_path)


class TestChunking:
    def _meta(self) -> dict:
        return {**REQUIRED_META, "source_file": "sample.txt"}

    def test_long_document_produces_multiple_chunks(self):
        text = ("Paragraph one about network segmentation.\n\n" * 60)
        chunks = chunk_document(text, self._meta())
        assert len(chunks) > 1
        assert chunks[0]["total_chunks"] == len(chunks)

    def test_chunk_size_respected_with_overlap(self):
        text = "-".join(f"token{i:03d}" for i in range(400))  # ~2800 chars, no split points
        chunks = chunk_document(text, self._meta())
        assert all(len(c["chunk_text"]) <= 1000 for c in chunks)
        assert len(chunks) >= 3
        # Consecutive chunks must overlap (start of chunk n+1 appears in chunk n).
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt["chunk_text"][:50] in prev["chunk_text"] + nxt["chunk_text"][:50] \
                or prev["chunk_text"][-50:] in nxt["chunk_text"]

    def test_metadata_completeness_in_every_chunk(self):
        text = "Line about PCI-DSS.\n\n" * 80
        chunks = chunk_document(text, self._meta())
        for i, c in enumerate(chunks):
            assert c["chunk_text"]
            assert c["chunk_index"] == i
            assert c["total_chunks"] == len(chunks)
            assert c["source_file"] == "sample.txt"
            assert c["source_type"] == "policy"
            assert c["business_unit"] == "finance"
            assert c["asset_criticality"] == "high"
            assert c["compliance_scope"] == "PCI-DSS"
            assert c["authority_level"] == "mandatory"

    def test_chunk_id_determinism_and_format(self):
        meta = self._meta()
        c1 = chunk_document("Some text about firewalls.\n\n" * 40, meta)
        c2 = chunk_document("Some text about firewalls.\n\n" * 40, meta)
        assert [c["chunk_id"] for c in c1] == [c["chunk_id"] for c in c2]
        for c in c1:
            assert len(c["chunk_id"]) == 16
            int(c["chunk_id"], 16)  # valid hex
        # Different source_file -> different ids
        other = chunk_document("Some text about firewalls.\n\n" * 40,
                               {**meta, "source_file": "other.txt"})
        assert c1[0]["chunk_id"] != other[0]["chunk_id"]

    def test_chunk_id_matches_spec_formula(self):
        assert _chunk_id("doc.txt", 3) == __import__("hashlib").sha256(
            b"doc.txt_3"
        ).hexdigest()[:16]

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            chunk_document("   ", self._meta())

    def test_missing_source_file_raises(self):
        with pytest.raises(KeyError):
            chunk_document("text", {k: v for k, v in REQUIRED_META.items()})


class TestIngestAll:
    def test_flattens_all_sample_documents(self):
        from org_rag_phase1.config import RAW_DIR

        chunks = ingest_all(RAW_DIR)
        files = {c["source_file"] for c in chunks}
        assert files == {"payment_policy.txt", "asset_inventory.txt", "network_topology.txt"}
        assert all(c["chunk_text"] for c in chunks)
        assert all("chunk_id" in c and "chunk_index" in c for c in chunks)
