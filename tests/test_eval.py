"""Tests for src/eval.py — the recon metrics harness.

Fully offline: metric math is tested against hand-built synthetic snapshots, and
the RAG-dependent snapshot path uses the isolated temp Chroma fixture. No live
probing, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from org_rag_phase1.src.agents.recon import DEFAULT_PORTS, Finding, ReconAgent
from org_rag_phase1.src.eval import (
    CRITICALITY_WEIGHTS,
    GroundTruthTarget,
    compute_metrics,
    load_ground_truth,
    snapshot_target,
)


def _finding(target, port, open_: bool = False) -> Finding:
    category = "port_open" if open_ else "port_filtered"
    detail = f"tcp/{port} open" if open_ else f"tcp/{port} filtered/closed"
    return Finding(target, category, detail, "info", "2026-09-16T00:00:00+00:00")


def _snapshot(arm, target, *, planned, executed=None, findings=None, restricted=False,
              criticality="high", context_n=3, latency=12.5, probe_latency=40.0,
              repeat_index=0, role="Some role", **extra):
    executed = executed if executed is not None else planned
    findings = findings if findings is not None else [_finding(target, p) for p in executed]
    return {
        "arm": arm, "target": target, "criticality": criticality,
        "context": {"asset": [{"source_file": "asset_inventory.txt"}] * context_n,
                    "topology": [], "policy": []},
        "context_latency_ms": latency, "role": role,
        "planned": planned, "executed": executed, "restricted": restricted,
        "findings": findings, "probe_latency_ms": probe_latency,
        "repeat_index": repeat_index, **extra,
    }


GT = {
    "10.0.1.5": GroundTruthTarget("10.0.1.5", role="Payment gateway",
                                  criticality="high",
                                  expected_sources=["asset_inventory.txt"],
                                  expert_ports=[443, 22, 80],
                                  listening_ports={443}),
    "10.0.2.10": GroundTruthTarget("10.0.2.10", role="Internal wiki",
                                   criticality="low",
                                   expected_sources=["asset_inventory.txt"],
                                   expert_ports=[80, 443],
                                   listening_ports={80}),
}


class TestGroundTruth:
    def test_load_parses_profile(self, tmp_path):
        p = tmp_path / "gt.json"
        p.write_text('{"10.0.1.5": {"role": "Payment gateway", '
                     '"expert_ports": [443], "listening_ports": [443, 443], '
                     '"expected_sources": ["x.txt"]}}')
        profiles = load_ground_truth(p)
        assert profiles["10.0.1.5"].expert_ports == [443]
        assert profiles["10.0.1.5"].listening_ports == {443}

    def test_none_returns_empty(self):
        assert load_ground_truth(None) == {}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ground_truth("/no/such/file.json")


@pytest.fixture(scope="module")
def indexed_eval(tmp_path_factory, embedder):
    from org_rag_phase1.src import index as index_mod

    root = tmp_path_factory.mktemp("eval")
    index_mod.CHROMA_PATH = str(root / "chroma_db")
    index_mod.get_embedder = lambda: embedder
    index_mod._client = None
    index_mod._collection = None
    from org_rag_phase1.config import RAW_DIR
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all
    index_chunks(ingest_all(RAW_DIR), reset=True)
    yield SimpleNamespace()
    index_mod._client = None
    index_mod._collection = None


class TestSnapshotTarget:
    def test_aware_arm_reads_context_and_plans(self, indexed_eval):
        agent = ReconAgent(targets=["10.0.1.5"], context_enabled=True)
        snap = snapshot_target(agent, "10.0.1.5", probe=False)
        assert snap["arm"] == "aware"
        assert snap["context_latency_ms"] > 0
        assert len(snap["planned"]) == len(DEFAULT_PORTS)
        assert snap["executed"] == []
        assert snap["findings"] == []

    def test_blind_arm_has_empty_context(self, indexed_eval):
        agent = ReconAgent(targets=["10.0.1.5"], context_enabled=False)
        snap = snapshot_target(agent, "10.0.1.5", probe=False)
        assert snap["arm"] == "blind"
        assert all(not v for v in snap["context"].values())
        assert snap["role"] == "unknown role"

    def test_probe_mode_records_findings(self, indexed_eval):
        agent = ReconAgent(targets=["127.0.0.1"], context_enabled=False)
        snap = snapshot_target(agent, "127.0.0.1", probe=True)
        assert len(snap["executed"]) == len(DEFAULT_PORTS)
        assert len(snap["findings"]) == len(DEFAULT_PORTS)
        assert all(f.category in {"port_open", "port_filtered"} for f in snap["findings"])
        assert "probe_latency_ms" in snap

    def test_guardrail_block_sets_restricted(self, indexed_eval):
        agent = ReconAgent(targets=["8.8.8.8"], context_enabled=True)
        snap = snapshot_target(agent, "8.8.8.8", probe=True)
        assert snap["restricted"] is True
        assert snap["findings"][0].category == "error"


class TestMetricsAwareBlind:
    def test_compute_full_with_profile(self):
        aware = [
            _snapshot("aware", "10.0.1.5", role="Payment gateway",
                      planned=[443, 22, 80], executed=[443, 22],
                      findings=[_finding("10.0.1.5", 443, open_=True),
                                _finding("10.0.1.5", 22)]),
            _snapshot("aware", "10.0.2.10", role="Internal wiki",
                      planned=[80, 443],
                      findings=[_finding("10.0.2.10", 80, open_=True),
                                _finding("10.0.2.10", 443)]),
        ]
        blind = [
            _snapshot("blind", "10.0.1.5", planned=[22, 80, 443], findings=[]),
            _snapshot("blind", "10.0.2.10", planned=[22, 80, 443], findings=[]),
        ]
        report = compute_metrics(aware + blind, GT, numbered=False)

        assert report["A1 context_hit_rate"] == 1.0
        assert report["A2 correct_source_rate"] == 1.0
        assert report["A3 context_precision"] == 1.0
        assert report["A4 role_correctness"] == 1.0
        assert report["B1 plan_precision"] == 1.0
        assert report["B2 plan_recall"] == 1.0
        assert report["B3 plan_order_correlation"] == 1.0
        assert report["B4 context_sensitivity"] == 1.0        # every plan differs
        assert report["B5 planned_ports_per_target"] == 2.5
        assert report["C1 probe_coverage"] == pytest.approx(0.833, abs=0.001)  # t1 executed 2/3
        assert report["C2 port_state_accuracy"] == 1.0
        assert report["C3 false_open_rate"] == 0.0
        assert report["C4 false_closed_rate"] == 0.0
        assert report["C7 probe_footprint"] == 4
        assert report["D1 finding_completeness"] == 1.0
        assert report["D2 schema_validity"] == 1.0
        assert report["D3 finding_validity"] == 1.0
        assert report["D4 traceability"] == 1.0
        assert report["E1 publish_success"] is None           # not published in this run
        assert report["R1 plan_difference_rate"] == 1.0
        assert report["R2 recall_delta_aware"] == 0.0         # recall equal in both arms
        assert report["R4 risk_weighted_yield_delta"] == 4.0  # 3*1 + 1*1 (aware) - 0 (blind)
        assert report["R6 consistency"] is None

    def test_no_profile_degrades_gracefully(self):
        aware = [_snapshot("aware", "10.0.1.5", planned=[443, 22, 80, 3389, 8443])]
        blind = [_snapshot("blind", "10.0.1.5", planned=[22, 80, 443, 3389, 8443])]
        report = compute_metrics(aware + blind, {})
        assert report["A1 context_hit_rate"] == 1.0
        assert report["A2 correct_source_rate"] is None
        assert report["B2 plan_recall"] is None
        assert report["C2 port_state_accuracy"] is None
        assert report["R2 recall_delta_aware"] is None
        assert report["R1 plan_difference_rate"] == 1.0

    def test_repeats_produce_consistency(self):
        first = _snapshot("aware", "t1", planned=[22, 80], repeat_index=0)
        repeat_same = _snapshot("aware", "t1", planned=[22, 80], repeat_index=1)
        repeat_diff = _snapshot("aware", "t2", planned=[80, 22], repeat_index=0)
        blind = _snapshot("blind", "t1", planned=[80, 22])
        report = compute_metrics([first, repeat_same, repeat_diff, blind], {})
        assert report["R6 consistency"] == 1.0

    def test_numbered_keys(self):
        report = compute_metrics(
            [_snapshot("aware", "t1", planned=[22]), _snapshot("blind", "t1", planned=[22])],
            {}, numbered=True)
        assert "A1" in report and "R1" in report and "B2" in report
        assert "A2 correct_source_rate" not in report