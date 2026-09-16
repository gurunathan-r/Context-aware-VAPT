"""Tests for src/retrieve.py: where-builder, filter enforcement, fallback, format."""

from __future__ import annotations

import pytest

from org_rag_phase1.src.index import index_chunks
from org_rag_phase1.src.ingest import ingest_all
from org_rag_phase1.src.retrieve import (
    _build_where,
    format_context_for_prompt,
    retrieve_org_context,
    retrieve_with_fallback,
)


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, embedder, request):
    """Module-scoped indexed env (shared across these tests for speed)."""
    from types import SimpleNamespace

    from org_rag_phase1.src import index as index_mod

    root = tmp_path_factory.mktemp("retrieve")
    chroma_dir = root / "chroma_db"
    chroma_dir.mkdir()

    original = (index_mod.CHROMA_PATH, index_mod.get_embedder)
    index_mod.CHROMA_PATH = str(chroma_dir)
    index_mod.get_embedder = lambda: embedder
    index_mod._client = None
    index_mod._collection = None

    index_chunks(ingest_all(), reset=True)

    yield SimpleNamespace(chroma_dir=chroma_dir)

    index_mod.CHROMA_PATH, index_mod.get_embedder = original
    index_mod._client = None
    index_mod._collection = None


class TestBuildWhere:
    def test_empty_returns_none(self):
        assert _build_where({}) is None

    def test_single_value_becomes_eq(self):
        assert _build_where({"source_type": "asset"}) == {"source_type": {"$eq": "asset"}}

    def test_list_value_becomes_in(self):
        assert _build_where({"compliance_scope": ["PCI-DSS", "GDPR"]}) == {
            "compliance_scope": {"$in": ["PCI-DSS", "GDPR"]}
        }

    def test_multiple_fields_combine_with_and(self):
        where = _build_where({"a": 1, "b": "x"})
        assert where == {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": "x"}}]}

    def test_explicit_operator_passes_through(self):
        where = _build_where({"asset_criticality": {"$in": ["high", "medium"]}})
        assert where == {"asset_criticality": {"$in": ["high", "medium"]}}

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="unsupported operator"):
            _build_where({"f": {"$regex": "x"}})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            _build_where({"f": []})


class TestRetrieveOrgContext:
    def test_returns_requested_keys(self, indexed):
        results = retrieve_org_context("compliance requirements for payment systems")
        assert results, "expected non-empty results"
        expected = {
            "chunk_text", "source_type", "business_unit", "asset_criticality",
            "compliance_scope", "authority_level", "source_file",
            "chunk_id", "chunk_index", "total_chunks", "distance",
        }
        assert expected.issubset(results[0].keys())

    def test_payment_query_hits_policy_with_pci_dss(self, indexed):
        results = retrieve_org_context("compliance requirements for payment systems")
        assert any(
            r["source_type"] == "policy" and r["compliance_scope"] == "PCI-DSS"
            for r in results
        )

    def test_top_k_respected(self, indexed):
        assert len(retrieve_org_context("firewall DMZ rules", top_k=2)) == 2

    def test_filter_enforcement(self, indexed):
        results = retrieve_org_context(
            "business criticality of 10.0.1.5", filters={"source_type": "asset"}
        )
        assert results
        assert all(r["source_type"] == "asset" for r in results)

    def test_in_filter(self, indexed):
        results = retrieve_org_context(
            "regulatory requirements",
            filters={"compliance_scope": {"$in": ["PCI-DSS", "GDPR"]}},
        )
        assert results
        assert all(r["compliance_scope"] in {"PCI-DSS", "GDPR"} for r in results)

    def test_no_match_filter_returns_empty(self, indexed):
        assert retrieve_org_context("anything", filters={"source_type": "intel"}) == []

    def test_empty_query_raises(self, indexed):
        with pytest.raises(ValueError, match="empty query"):
            retrieve_org_context("   ")

    def test_distances_sorted_ascending(self, indexed):
        results = retrieve_org_context("payment gateway criticality")
        dists = [r["distance"] for r in results]
        assert dists == sorted(dists)


class TestRetrieveWithFallback:
    def test_filter_hit_marks_no_fallback(self, indexed):
        results = retrieve_with_fallback(
            "payment gateway", filters={"source_type": "asset"}, min_results=1
        )
        assert results and all(r["filtered_fallback"] is False for r in results)

    def test_filter_miss_falls_back(self, indexed):
        results = retrieve_with_fallback(
            "PCI-DSS requirements", filters={"source_type": "intel"}, min_results=1
        )
        assert results
        assert all(r["filtered_fallback"] is True for r in results)
        assert any("PCI-DSS" in r["chunk_text"] for r in results)

    def test_fallback_result_has_full_schema(self, indexed):
        results = retrieve_with_fallback(
            "anything", filters={"business_unit": "nonexistent"}, min_results=1
        )
        assert "chunk_text" in results[0] and "distance" in results[0]


class TestFormatContextForPrompt:
    def test_block_structure(self, indexed):
        results = retrieve_org_context("payment compliance")
        block = format_context_for_prompt(results)
        assert block.startswith("[ORG CONTEXT]")
        assert block.rstrip().endswith("[END ORG CONTEXT]")
        assert "Source: payment_policy.txt | Type: policy | Criticality: high" in block
        assert "Authority: mandatory | Compliance: PCI-DSS" in block

    def test_chunk_text_included(self, indexed):
        results = retrieve_org_context("payment compliance", top_k=1)
        block = format_context_for_prompt(results)
        assert results[0]["chunk_text"] in block

    def test_empty_results_block(self):
        block = format_context_for_prompt([])
        assert "No organizational context" in block
        assert block.startswith("[ORG CONTEXT]")
