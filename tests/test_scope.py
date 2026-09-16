"""Tests for src/scope.py + bug-bounty scope gating in the Recon Agent.

Fully offline: the network fetch is mocked, index fixtures use the isolated
temp Chroma store, and recon gating monkeypatches check_scope/probe_port so no
live probing ever happens.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from org_rag_phase1.src.scope import (
    _domain_matches,
    _ip_matches,
    _parse_fields,
    _normalize_host,
    check_scope,
    fetch_security_txt,
    host_is_ip,
    index_scope_docs,
    parse_scope_entries,
    save_scope_doc,
    target_in_scope,
)

SAMPLE_POLICY = """Contact: mailto:security@example.com
Canonical: https://example.com/.well-known/security.txt
Expires: 2030-12-31T23:59:59.000Z

In-scope assets:
- *.example.com
- api.example.com
- 10.20.0.0/16

Out of scope: any host outside the above.
"""


@pytest.fixture(scope="module")
def scope_index(tmp_path_factory, embedder):
    """Isolated Chroma store pre-indexed with one source_type=scope document."""
    from org_rag_phase1.src import index as index_mod

    root = tmp_path_factory.mktemp("scope")
    chroma_dir = root / "chroma_db"
    scope_dir = root / "scope"
    chroma_dir.mkdir()
    scope_dir.mkdir()

    original = (index_mod.CHROMA_PATH, index_mod.get_embedder)
    index_mod.CHROMA_PATH = str(chroma_dir)
    index_mod.get_embedder = lambda: embedder
    index_mod._client = None
    index_mod._collection = None

    save_scope_doc("example.com", {
        "domain": "example.com",
        "url": "https://example.com/.well-known/security.txt",
        "text": SAMPLE_POLICY,
        "parsed": _parse_fields(SAMPLE_POLICY),
    }, scope_dir=str(scope_dir))
    index_scope_docs(str(scope_dir), reset=True)

    yield SimpleNamespace(chroma_dir=chroma_dir, scope_dir=scope_dir)

    index_mod.CHROMA_PATH, index_mod.get_embedder = original
    index_mod._client = None
    index_mod._collection = None


# ---------------------------------------------------------------------------
# Pure parsing / matching
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_extracts_domains_wildcards_cidrs(self):
        entries = parse_scope_entries(SAMPLE_POLICY)
        assert "example.com" in entries
        assert "api.example.com" in entries
        assert "10.20.0.0/16" in entries

    def test_parse_skips_field_metadata_lines(self):
        entries = parse_scope_entries("Expires: 2030-12-31T23:59:59.000Z\n" + SAMPLE_POLICY)
        assert all("2030" not in e and "000z" not in e for e in entries)

    def test_parse_fields_rfc9116(self):
        fields = _parse_fields("Contact: mailto:a@b.com\nExpires: 2030-01-01\n")
        assert fields["Contact"] == "mailto:a@b.com"
        assert fields["Expires"] == "2030-01-01"

    def test_normalize_strips_port_and_case(self):
        assert _normalize_host("Api.Example.com:443") == "api.example.com"

    def test_host_is_ip(self):
        assert host_is_ip("10.0.1.5")
        assert host_is_ip("2001:db8::1")
        assert not host_is_ip("api.example.com")


class TestMatching:
    def test_domain_exact(self):
        assert _domain_matches("example.com", "example.com")
        assert _domain_matches("EXAMPLE.com", "example.com")

    def test_subdomain_in_scope(self):
        assert _domain_matches("api.example.com", "example.com")

    def test_wildcard_entry_apex(self):
        assert _domain_matches("example.com", "*.example.com")

    def test_unrelated_domain_out(self):
        assert not _domain_matches("other.com", "example.com")

    def test_ip_in_cidr(self):
        assert _ip_matches("10.20.3.4", "10.20.0.0/16")

    def test_ip_outside_cidr(self):
        assert not _ip_matches("10.21.0.1", "10.20.0.0/16")

    def test_target_in_scope_hostname_and_ip(self):
        texts = [SAMPLE_POLICY]
        assert target_in_scope("api.example.com", texts)
        assert target_in_scope("deep.sub.example.com", texts)
        assert target_in_scope("10.20.7.7", texts)
        assert not target_in_scope("other.com", texts)
        assert not target_in_scope("10.21.0.1", texts)


# ---------------------------------------------------------------------------
# network fetch (mocked) + persistence
# ---------------------------------------------------------------------------

class TestFetch:
    def _fake_get(self, responses):
        def fake(url, timeout=None, headers=None, allow_redirects=True):
            body = responses.get(url)
            if body is None:
                class R:
                    status_code = 404
                    text = ""
                return R()
            class R:
                status_code = 200
                text = body
            return R()
        return fake

    def test_fetches_well_known_https(self, monkeypatch):
        import org_rag_phase1.src.scope as scope_mod
        policy = "Canonical: https://a.com/.well-known/security.txt\nContact: mailto:s@a.com\n"
        monkeypatch.setattr(scope_mod.requests, "get", self._fake_get({
            "https://example.com/.well-known/security.txt": policy,
        }))
        payload = fetch_security_txt("example.com")
        assert payload["url"] == "https://example.com/.well-known/security.txt"
        assert payload["parsed"]["Contact"] == "mailto:s@a.com"

    def test_falls_back_to_legacy_path_then_http(self, monkeypatch):
        import org_rag_phase1.src.scope as scope_mod
        policy = "Contact: mailto:s@a.com\n"
        monkeypatch.setattr(scope_mod.requests, "get", self._fake_get({
            "http://example.com/security.txt": policy,
        }))
        payload = fetch_security_txt("example.com")
        assert payload["url"] == "http://example.com/security.txt"

    def test_no_policy_returns_none(self, monkeypatch):
        import org_rag_phase1.src.scope as scope_mod
        monkeypatch.setattr(scope_mod.requests, "get", self._fake_get({}))
        assert fetch_security_txt("example.com") is None

    def test_ip_domain_rejected(self):
        with pytest.raises(ValueError):
            fetch_security_txt("10.0.1.5")


class TestSaveAndIndex:
    def test_save_writes_md_and_sidecar(self, tmp_path):
        paths = save_scope_doc("example.com", {
            "domain": "example.com",
            "url": "https://example.com/security.txt",
            "text": SAMPLE_POLICY,
            "parsed": _parse_fields(SAMPLE_POLICY),
        }, scope_dir=str(tmp_path / "scope"))
        assert paths["doc_path"].endswith("scope_example.com.md")
        assert paths["meta_path"].endswith("scope_example.com.md.meta.json")
        meta = json.loads(open(paths["meta_path"], encoding="utf-8").read())
        assert meta["source_type"] == "scope"
        assert meta["authority_level"] == "mandatory"

    def test_index_and_check_scope_round_trip(self, scope_index):
        """The agent-facing contract: published scope makes hosts in/out-classified."""
        verdict = check_scope("api.example.com")
        assert verdict["status"] == "in_scope"
        assert any(r["source_file"].startswith("scope_") for r in verdict["records"])

        verdict_ip = check_scope("10.20.3.4")
        assert verdict_ip["status"] in {"in_scope", "out_of_scope"}

        verdict_out = check_scope("other-org.com")
        assert verdict_out["status"] == "out_of_scope"


# ---------------------------------------------------------------------------
# Recon Agent gating (no live probing)
# ---------------------------------------------------------------------------

class TestReconScopeGating:
    def test_out_of_scope_blocks_all_probing(self, monkeypatch):
        import org_rag_phase1.src.agents.recon as recon_mod

        called = []
        monkeypatch.setattr(recon_mod, "check_scope",
                            lambda target: {"status": "out_of_scope", "records": [
                                {"source_file": "scope_example.com.md"}]})
        monkeypatch.setattr(recon_mod, "probe_port",
                            lambda host, port, timeout=1.0: called.append(port))

        agent = recon_mod.ReconAgent(targets=["localhost"])
        findings = agent.run_recon("localhost")

        assert called == []                     # zero probes executed
        assert len(findings) == 1               # single blocking record
        assert findings[0].extra["scope_status"] == "out_of_scope"
        assert findings[0].extra["probes_planned"] == 0

    def test_in_scope_proceeds_and_tags_context(self, monkeypatch):
        import org_rag_phase1.src.agents.recon as recon_mod

        def fake_probe(host, port, timeout=1.0):
            return recon_mod.Finding(host, "port_open", f"tcp/{port} open",
                                     "info", "now")

        monkeypatch.setattr(recon_mod, "check_scope",
                            lambda target: {"status": "in_scope", "records": [
                                {"source_file": "scope_example.com.md"}]})
        monkeypatch.setattr(recon_mod, "probe_port", fake_probe)

        agent = recon_mod.ReconAgent(targets=["app.example.com"], allow_public=True)
        findings = agent.run_recon("app.example.com")

        assert findings[0].extra["scope_status"] == "in_scope"
        categories = {f.category for f in findings}
        assert "port_open" in categories          # probes actually ran
        assert any(f.extra.get("scope_status") == "in_scope" for f in findings)

    def test_no_policy_preserves_legacy_behavior(self, monkeypatch):
        """No publication => guardrail-only path (private/lab targets still probe)."""
        import org_rag_phase1.src.agents.recon as recon_mod

        monkeypatch.setattr(recon_mod, "check_scope",
                            lambda target: {"status": "no_policy", "records": []})

        agent = recon_mod.ReconAgent(targets=["localhost"])
        findings = agent.run_recon("localhost")

        assert findings[0].category == "info"     # context record first, as before
        assert {"port_open", "port_filtered"} & {f.category for f in findings}