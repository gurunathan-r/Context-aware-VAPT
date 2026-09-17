"""Tests for src/agents/context_agent.py.

Runs fully offline using the isolated Chroma test environment fixture.
"""

from __future__ import annotations

import pytest

from org_rag_phase1.src.agents.context_agent import (
    EnrichedVulnerability,
    OrgContextAgent,
)


@pytest.fixture()
def sample_finding():
    return {
        "id": "TEST-001",
        "cve": "CVE-2023-38606",
        "title": "Payment Gateway SQLi",
        "target": "10.0.1.5",
        "port": 443,
        "service": "https",
        "cvss_score": 7.5,
        "severity": "HIGH",
        "description": "SQL injection in payment gateway",
        "attack_vector": "Network",
        "assumed_origin": "Internet via DMZ proxy",
    }


class TestOrgContextAgent:
    def test_blind_agent_has_empty_context(self, sample_finding):
        agent = OrgContextAgent(context_enabled=False)
        ctx = agent.gather_context_for_finding("10.0.1.5", 443)
        assert ctx["asset"] == []
        assert ctx["topology"] == []
        assert ctx["policy"] == []

    def test_dead_end_detection_fw014(self):
        agent = OrgContextAgent()
        # Port 8080 on 10.0.1.5 should be flagged as dead end due to FW-014
        finding = {
            "id": "TEST-DEAD",
            "target": "10.0.1.5",
            "port": 8080,
            "assumed_origin": "Internal segment",
        }
        is_dead_end, reason = agent.analyze_reachability(finding, [])
        assert is_dead_end is True
        assert "FW-014" in reason

    def test_reachable_path_payment_443(self):
        agent = OrgContextAgent()
        finding = {
            "id": "TEST-OK",
            "target": "10.0.1.5",
            "port": 443,
            "assumed_origin": "Internet via DMZ proxy",
        }
        is_dead_end, _ = agent.analyze_reachability(finding, [])
        assert is_dead_end is False

    def test_risk_score_calculation(self):
        agent = OrgContextAgent()
        # High criticality (2.0) * PCI-DSS (1.5) * DMZ (1.3) * Reachable (1.0)
        score = agent.calculate_risk_score(
            cvss=7.5,
            criticality="high",
            compliance="PCI-DSS",
            exposure="dmz",
            is_dead_end=False,
        )
        assert score == 29.25

    def test_dead_end_heavily_discounted(self):
        agent = OrgContextAgent()
        score = agent.calculate_risk_score(
            cvss=9.8,
            criticality="high",
            compliance="PCI-DSS",
            exposure="internal",
            is_dead_end=True,
        )
        # 9.8 * 2.0 * 1.5 * 0.8 * 0.1 = 2.35
        assert score == 2.35

    def test_process_findings_sorting(self, sample_finding):
        agent = OrgContextAgent(context_enabled=True, use_llm=False)
        findings = [
            sample_finding,
            {
                "id": "TEST-DEAD",
                "cve": "CVE-2022-26134",
                "title": "Admin Console RCE",
                "target": "10.0.1.5",
                "port": 8080,
                "service": "http-admin",
                "cvss_score": 9.8,
                "severity": "CRITICAL",
                "assumed_origin": "Internal segment",
            },
        ]
        blind, aware = agent.process_findings(findings)

        # In blind arm, 9.8 outranks 7.5
        assert blind[0]["id"] == "TEST-DEAD"
        assert blind[1]["id"] == "TEST-001"

        # In aware arm, 7.5 (reachable, PCI-DSS) outranks 9.8 (dead end)
        assert aware[0].id == "TEST-001"
        assert aware[1].id == "TEST-DEAD"
        assert aware[1].is_dead_end is True

    def test_generate_markdown_reports(self, sample_finding):
        agent = OrgContextAgent(context_enabled=True, use_llm=False)
        blind, aware = agent.process_findings([sample_finding])
        b_md, a_md = agent.generate_markdown_reports(blind, aware)
        assert "Vulnerability Assessment Report (Baseline: Context-Blind)" in b_md
        assert "Context-Aware Vulnerability Prioritization Report" in a_md
        assert "ACTIONABLE" in a_md

    def test_parse_llm_json_variants(self):
        agent = OrgContextAgent(use_llm=False)

        # Pure JSON
        assert agent._parse_llm_json('{"is_dead_end": true}') == {"is_dead_end": True}

        # Code block markdown
        md_text = 'Here is the response:\n```json\n{"is_dead_end": false, "score": 25.0}\n```\nDone.'
        assert agent._parse_llm_json(md_text) == {"is_dead_end": False, "score": 25.0}

        # Reasoning / think tags (e.g. Qwen / DeepSeek)
        think_text = '<think>I should evaluate the firewall rules...</think>{"is_dead_end": true, "reason": "FW-014"}'
        assert agent._parse_llm_json(think_text) == {"is_dead_end": True, "reason": "FW-014"}

        # Empty / malformed
        assert agent._parse_llm_json("") is None
        assert agent._parse_llm_json("not valid json at all") is None

    def test_enrich_finding_with_llm_mock(self, sample_finding, monkeypatch):
        agent = OrgContextAgent(context_enabled=True, use_llm=True)

        mock_llm_output = {
            "is_dead_end": False,
            "dead_end_reason": "Reachable via DMZ reverse proxy",
            "asset_role": "Production Payment Gateway",
            "business_unit": "finance",
            "asset_criticality": "high",
            "compliance_scope": "PCI-DSS",
            "context_risk_score": 32.5,
            "adjusted_priority": "P1 - Immediate",
            "remediation_sla_days": 30,
            "business_impact": "Direct credit card transaction disruption",
            "citations": ["asset_inventory.txt", "payment_policy.txt"],
            "rationale": "Autonomous LLM reasoning confirmed critical PCI impact.",
        }

        monkeypatch.setattr(agent, "_call_llm_reasoning", lambda finding, ctx, cites: mock_llm_output)

        enriched = agent.enrich_finding(sample_finding)
        assert enriched.id == "TEST-001"
        assert enriched.is_dead_end is False
        assert enriched.context_risk_score == 32.5
        assert enriched.adjusted_priority == "P1 - Immediate"
        assert enriched.business_unit == "finance"
        assert enriched.remediation_sla_days == 30
        assert "payment_policy.txt" in enriched.citations
        assert "Autonomous LLM reasoning" in enriched.rationale

    def test_llm_failure_falls_back_to_analyzer(self, sample_finding, monkeypatch):
        agent = OrgContextAgent(context_enabled=True, use_llm=True)
        # Force LLM call to return None
        monkeypatch.setattr(agent, "_call_llm_reasoning", lambda finding, ctx, cites: None)

        enriched = agent.enrich_finding(sample_finding)
        # Should still be enriched via fallback context analyzer
        assert enriched.id == "TEST-001"
        assert isinstance(enriched.context_risk_score, float)
        assert enriched.context_risk_score > 0


