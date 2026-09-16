"""Tests for src/validate.py using a tiny synthetic, fully offline test set."""

from __future__ import annotations

import json

import pytest

from org_rag_phase1.src.validate import (
    PASS_THRESHOLD,
    load_test_queries,
    print_report,
    run_validation,
)


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, embedder):
    """Module-scoped isolated index shared by these tests (same as test_retrieve)."""
    from types import SimpleNamespace

    from org_rag_phase1.src import index as index_mod
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    root = tmp_path_factory.mktemp("validate")
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


@pytest.fixture()
def synthetic_queries():
    """Hand-written queries that must pass against the sample corpus."""
    return [
        {
            "query": "What are the compliance requirements for payment systems?",
            "filters": None,
            "expected_source_files": ["payment_policy.txt"],
            "expected_keywords": ["PCI-DSS", "cardholder"],
        },
        {
            "query": "Business criticality of asset 10.0.1.5",
            "filters": {"source_type": "asset"},
            "expected_source_files": ["asset_inventory.txt"],
            "expected_keywords": ["payment gateway"],
        },
        {
            "query": "Firewall restrictions on DMZ access",
            "filters": {"source_type": "topology"},
            "expected_source_files": ["network_topology.txt"],
            "expected_keywords": ["DMZ", "FW-014"],
        },
    ]


class TestLoadTestQueries:
    def test_loads_real_test_set(self):
        queries = load_test_queries()
        assert len(queries) == 5
        required = {"query", "filters", "expected_source_files", "expected_keywords"}
        assert all(required.issubset(q) for q in queries)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_test_queries(tmp_path / "nope.json")

    def test_invalid_structure_raises(self, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps([{"query": "incomplete"}]))
        with pytest.raises(ValueError, match="must be an object"):
            load_test_queries(p)

    def test_non_list_raises(self, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_test_queries(p)


class TestRunValidation:
    def test_report_structure(self, indexed, synthetic_queries):
        report = run_validation(synthetic_queries, top_k=3)
        assert report["total_queries"] == 3
        assert 0.0 <= report["source_file_hit_rate"] <= 1.0
        assert 0.0 <= report["keyword_hit_rate"] <= 1.0
        assert len(report["per_query_results"]) == 3
        entry = report["per_query_results"][0]
        assert {"query", "hit_source_file", "hit_keyword",
                "retrieved_source_files"}.issubset(entry)

    def test_perfect_score_on_synthetic_set(self, indexed, synthetic_queries):
        report = run_validation(synthetic_queries, top_k=3)
        assert report["source_file_hit_rate"] == 1.0
        assert report["keyword_hit_rate"] == 1.0

    def test_hit_rates_are_averages(self, indexed):
        queries = [
            {"query": "payment compliance", "filters": None,
             "expected_source_files": ["payment_policy.txt"],
             "expected_keywords": ["PCI-DSS"]},
            {"query": "payment compliance", "filters": None,
             "expected_source_files": ["definitely_absent.txt"],
             "expected_keywords": []},
        ]
        report = run_validation(queries, top_k=3)
        assert report["source_file_hit_rate"] == pytest.approx(0.5)
        assert report["keyword_hit_rate"] == 1.0  # empty keyword list counts as hit

    def test_empty_input_raises(self, indexed):
        with pytest.raises(ValueError, match="empty test query"):
            run_validation([])


class TestPrintReport:
    def test_prints_pass_and_thresholds(self, indexed, synthetic_queries, capsys):
        report = run_validation(synthetic_queries, top_k=3)
        print_report(report)
        out = capsys.readouterr().out
        assert "RETRIEVAL VALIDATION REPORT" in out
        assert "PASS" in out
        assert str(PASS_THRESHOLD) in out

    def test_prints_fail_on_misses(self, indexed, capsys):
        report = run_validation(
            [{"query": "zzz nothing matches this", "filters": None,
              "expected_source_files": ["nope.txt"], "expected_keywords": ["zzzz"]}],
            top_k=2,
        )
        print_report(report)
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "[MISS]" in out
