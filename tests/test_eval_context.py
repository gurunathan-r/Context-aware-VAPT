"""Tests for src/eval_context.py.

Verifies metric computations: Spearman rho, Kendall tau, NDCG@K,
dead-end precision/recall, and alert fatigue reduction.
"""

from __future__ import annotations

import pytest

from org_rag_phase1.src.agents.context_agent import EnrichedVulnerability
from org_rag_phase1.src.eval_context import (
    evaluate_prioritization,
    kendall_tau,
    ndcg_at_k,
    spearman_rho,
)


def test_spearman_rho_perfect():
    assert spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_spearman_rho_reversed():
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_kendall_tau_perfect():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_kendall_tau_reversed():
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_ndcg_at_k_perfect():
    gt = {
        "A": {"expert_priority_rank": 1},
        "B": {"expert_priority_rank": 2},
        "C": {"expert_priority_rank": 3},
    }
    score = ndcg_at_k(["A", "B", "C"], gt, k=3)
    assert score == 1.0


def test_evaluate_prioritization_end_to_end():
    blind_findings = [
        {"id": "V1", "cvss_score": 10.0, "severity": "CRITICAL"},
        {"id": "V2", "cvss_score": 7.0, "severity": "HIGH"},
    ]
    aware_findings = [
        EnrichedVulnerability(
            id="V2",
            cve="CVE-2",
            title="Vuln 2",
            target="10.0.1.5",
            port=443,
            service="https",
            cvss_score=7.0,
            raw_severity="HIGH",
            description="",
            is_dead_end=False,
            context_risk_score=20.0,
            adjusted_priority="P1 - Immediate",
            remediation_sla_days=30,
            citations=["doc1.txt"],
        ),
        EnrichedVulnerability(
            id="V1",
            cve="CVE-1",
            title="Vuln 1",
            target="10.0.3.20",
            port=445,
            service="smb",
            cvss_score=10.0,
            raw_severity="CRITICAL",
            description="",
            is_dead_end=True,
            context_risk_score=1.0,
            adjusted_priority="P4 - Low / Hygiene",
            remediation_sla_days=180,
            citations=["doc2.txt"],
        ),
    ]
    gt = {
        "V1": {
            "expert_priority_rank": 2,
            "business_impact": "none",
            "is_dead_end": True,
            "is_compliance_violation": False,
        },
        "V2": {
            "expert_priority_rank": 1,
            "business_impact": "critical",
            "is_dead_end": False,
            "is_compliance_violation": True,
        },
    }

    metrics = evaluate_prioritization(blind_findings, aware_findings, gt)

    # Blind inverted order (V1 then V2), so rho should be -1.0
    assert metrics["M1_ranking_alignment"]["spearman_rho"]["blind"] == -1.0
    # Aware matched expert order (V2 then V1), so rho should be 1.0
    assert metrics["M1_ranking_alignment"]["spearman_rho"]["aware"] == 1.0

    # Dead end detection
    assert metrics["M2_dead_end_detection"]["recall"]["blind"] == 0.0
    assert metrics["M2_dead_end_detection"]["recall"]["aware"] == 1.0

    # Citations
    assert metrics["M5_context_grounding_citation_pct"]["aware"] == 100.0

