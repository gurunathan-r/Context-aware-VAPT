"""Tests for the Recon Agent — no external network required.

Probes target 127.0.0.1 (loopback) and the closed test port 9 (discard); the
RAG side uses the real embedder with an isolated temp Chroma store.
"""

from __future__ import annotations

import json
import socket

import pytest

from org_rag_phase1.src.agents.recon import (
    CANDIDATE_PORTS,
    DEFAULT_PORTS,
    Finding,
    PORT_HINT_RULES,
    ReconAgent,
    _extract_role,
    check_target_allowed,
    plan_probes,
    probe_port,
)


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, embedder):
    """Module-scoped isolated index (same pattern as other test modules)."""
    from types import SimpleNamespace

    from org_rag_phase1.src import index as index_mod
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    root = tmp_path_factory.mktemp("recon")
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


class TestGuardrail:
    def test_private_ip_allowed(self):
        assert check_target_allowed("10.0.1.5") == "10.0.1.5"

    def test_public_ip_rejected(self):
        with pytest.raises(ValueError, match="not a private/lab address"):
            check_target_allowed("8.8.8.8")

    def test_public_ip_allowed_only_with_override(self):
        assert check_target_allowed("8.8.8.8", allow_public=True) == "8.8.8.8"

    def test_resolvable_private_hostname_allowed(self):
        ip = check_target_allowed("localhost")
        assert ip  # resolved (e.g. 127.0.0.1)

    def test_unresolvable_public_hostname_rejected(self):
        with pytest.raises(ValueError):
            check_target_allowed("this.host.does.not.exist.invalid")


class TestProbePlanning:
    def test_defaults_when_no_hints(self):
        assert plan_probes([]) == list(DEFAULT_PORTS)

    def test_https_hint_moves_443_first(self):
        context = [{"chunk_text": "Only HTTPS (tcp/443) is permitted to the gateway."}]
        assert plan_probes(context)[0] == 443

    def test_database_hint_adds_db_port(self):
        context = [{"chunk_text": "The HR database stores personnel records."}]
        assert 3389 in plan_probes(context)

    def test_database_hint_also_adds_mssql_1433(self):
        """Regression: the HR database is MSSQL on Windows — a "database" hint
        must plan tcp/1433 too, or expert recall (Q6) can never reach 1.0."""
        context = [{"chunk_text": "The HR database stores personnel records."}]
        plan = plan_probes(context)
        assert 1433 in plan
        assert plan.index(1433) < plan.index(80)   # hinted ports come first

    def test_planned_ports_stay_within_the_candidate_universe(self):
        """Whatever the hints say, the plan never leaves CANDIDATE_PORTS."""
        context = [{"chunk_text": "https ssh database sql 443 22 1433 3389 8443 80"}]
        assert set(plan_probes(context)) <= set(CANDIDATE_PORTS)

    def test_every_hint_port_is_in_the_candidate_universe(self):
        """The contract table can never hint a port the budget forbids."""
        assert all(port in CANDIDATE_PORTS for _, port in PORT_HINT_RULES)


class TestProbePort:
    def test_open_port_on_loopback(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            f = probe_port("127.0.0.1", port, timeout=1.0)
            assert f.category == "port_open"
        finally:
            srv.close()

    def test_closed_port_is_filtered(self):
        f = probe_port("127.0.0.1", 9, timeout=0.5)  # discard port usually closed
        if f.category == "port_open":  # skip if this machine actually runs discard
            pytest.skip("discard port open on this host")
        assert f.category == "port_filtered"


class TestRoleExtraction:
    def test_role_near_target_mention_wins(self):
        text = (
            "ASSET 10.0.1.5\nRole: Payment gateway\nstuff\n\n"
            "ASSET 10.0.2.10\nRole: Internal wiki\n\n"
            "ASSET 10.0.3.20\nRole: HR database\n"
        )
        assert _extract_role([{"chunk_text": text}], "10.0.3.20") == "HR database"
        assert _extract_role([{"chunk_text": text}], "10.0.2.10") == "Internal wiki"
        assert _extract_role([{"chunk_text": text}], "10.0.1.5") == "Payment gateway"

    def test_unknown_when_no_mention(self):
        assert _extract_role([{"chunk_text": "Role: Something"}], "10.9.9.9") == "unknown role"


class TestReconAgent:
    def test_run_recon_on_private_target(self, indexed):
        agent = ReconAgent(targets=["10.0.1.5"])
        findings = agent.run_recon("10.0.1.5")
        assert findings[0].category == "info"          # context record first
        assert findings[0].extra["context_chunks"] > 0  # RAG actually consulted
        # Every context-derived claim carries the documents that support it.
        assert findings[0].extra["sources"], "claims must be traceable to retrieved documents"
        categories = {f.category for f in findings}
        assert categories <= {"info", "port_open", "port_filtered", "error"}

    def test_public_target_skipped_not_crashed(self, indexed):
        agent = ReconAgent(targets=["8.8.8.8"])
        result = agent.run(index_findings=False)
        assert result["findings"]["8.8.8.8"][0].category == "error"

    def test_publish_and_retrieve_loop(self, indexed, tmp_path):
        """The core claim: findings written by the agent become retrievable RAG knowledge."""
        agent = ReconAgent(targets=["10.0.1.5"], intel_dir=tmp_path / "intel")
        findings = agent.run_recon("10.0.1.5")
        pub = agent.publish_findings({"10.0.1.5": findings}, index=True)

        assert (tmp_path / "intel").exists()
        report = open(pub["report_path"], encoding="utf-8").read()
        assert "# Recon Findings" in report and "10.0.1.5" in report

        meta = json.loads(open(pub["meta_path"], encoding="utf-8").read())
        assert meta["source_type"] == "intel"

        # The loop: retrieve the agent's own findings from the RAG.
        hits = __import__("org_rag_phase1.src.retrieve", fromlist=["retrieve_org_context"]) \
            .retrieve_org_context("recon findings port scan results 10.0.1.5", top_k=3)
        assert any(h["source_file"].startswith("recon_findings_") for h in hits)

    def test_publish_no_index_dry_run(self, indexed, tmp_path):
        agent = ReconAgent(targets=["10.0.1.5"], intel_dir=tmp_path / "intel")
        pub = agent.publish_findings({"10.0.1.5": [Finding("10.0.1.5", "info", "x", "info", "now")]},
                                     index=False)
        assert pub["chunks_indexed"] == 0
        assert (tmp_path / "intel").exists()


class TestFinding:
    def test_as_line_format(self):
        f = Finding("10.0.1.5", "port_open", "tcp/443 open", "info", "now", {"note": "HTTPS"})
        assert "port_open" in f.as_line() and "tcp/443 open" in f.as_line()
