"""Evaluation Harness for Context-Aware Vulnerability Prioritization.

Computes quantitative metrics comparing Context-Blind (CVSS-only) vs.
Context-Aware (Organizational RAG) reporting against an expert ground truth benchmark.

Metrics implemented:
  M1. Rank Correlation (Spearman rho, Kendall tau, NDCG@K) vs Expert Ground Truth
  M2. Dead-End Detection Performance (Precision, Recall, F1)
  M3. False Urgency Reduction (Alert Fatigue Mitigation Rate)
  M4. Compliance SLA Alignment Rate
  M5. Context Grounding / Citation Rate
  M6. Top-K Actionable Remediation Precision (ARP@K)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from org_rag_phase1.src.agents.context_agent import EnrichedVulnerability


def spearman_rho(pred_ranks: list[int], gt_ranks: list[int]) -> float:
    """Compute Spearman's rank correlation coefficient rho."""
    n = len(pred_ranks)
    if n < 2:
        return 1.0
    d_sq_sum = sum((p - g) ** 2 for p, g in zip(pred_ranks, gt_ranks))
    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n**2 - 1))
    return round(rho, 4)


def kendall_tau(pred_ranks: list[int], gt_ranks: list[int]) -> float:
    """Compute Kendall's rank correlation coefficient tau."""
    n = len(pred_ranks)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = pred_ranks[i] - pred_ranks[j]
            gt_diff = gt_ranks[i] - gt_ranks[j]
            if pred_diff * gt_diff > 0:
                concordant += 1
            elif pred_diff * gt_diff < 0:
                discordant += 1
    total_pairs = (n * (n - 1)) / 2
    tau = (concordant - discordant) / total_pairs
    return round(tau, 4)


def ndcg_at_k(items_order: list[str], ground_truth: dict[str, Any], k: int = 3) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    n = len(ground_truth)
    # Relevance is higher for top priority: rel = (max_rank - expert_rank + 1)
    relevances = {
        vid: (n - spec.get("expert_priority_rank", n) + 1)
        for vid, spec in ground_truth.items()
    }

    # Actual DCG@K
    dcg = 0.0
    for i, vid in enumerate(items_order[:k]):
        rel = relevances.get(vid, 0)
        dcg += rel / math.log2(i + 2)

    # Ideal DCG@K (sorted by true relevance descending)
    ideal_order = sorted(relevances.keys(), key=lambda vid: relevances[vid], reverse=True)
    idcg = 0.0
    for i, vid in enumerate(ideal_order[:k]):
        rel = relevances[vid]
        idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def evaluate_prioritization(
    blind_findings: list[dict[str, Any]],
    aware_findings: list[EnrichedVulnerability],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics comparing blind vs aware reporting."""
    vid_to_gt = ground_truth

    # --- 1. Ranking Metrics ---
    # Ranks for Blind arm (1-based)
    blind_order = [f["id"] for f in blind_findings]
    blind_pred_ranks = [i + 1 for i in range(len(blind_order))]
    blind_gt_ranks = [vid_to_gt[vid]["expert_priority_rank"] for vid in blind_order]

    # Ranks for Aware arm (1-based)
    aware_order = [f.id for f in aware_findings]
    aware_pred_ranks = [i + 1 for i in range(len(aware_order))]
    aware_gt_ranks = [vid_to_gt[vid]["expert_priority_rank"] for vid in aware_order]

    m1_blind_rho = spearman_rho(blind_pred_ranks, blind_gt_ranks)
    m1_aware_rho = spearman_rho(aware_pred_ranks, aware_gt_ranks)

    m1_blind_tau = kendall_tau(blind_pred_ranks, blind_gt_ranks)
    m1_aware_tau = kendall_tau(aware_pred_ranks, aware_gt_ranks)

    m1_blind_ndcg3 = ndcg_at_k(blind_order, ground_truth, k=3)
    m1_aware_ndcg3 = ndcg_at_k(aware_order, ground_truth, k=3)

    # --- 2. Dead-End Detection Metrics ---
    gt_dead_ends = {vid for vid, spec in vid_to_gt.items() if spec.get("is_dead_end")}
    aware_dead_ends = {f.id for f in aware_findings if f.is_dead_end}
    blind_dead_ends: set[str] = set()  # Blind arm has zero dead-end detection

    def calc_pr_f1(pred_set: set[str], true_set: set[str]) -> tuple[float, float, float]:
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    m2_blind_prec, m2_blind_rec, m2_blind_f1 = calc_pr_f1(blind_dead_ends, gt_dead_ends)
    m2_aware_prec, m2_aware_rec, m2_aware_f1 = calc_pr_f1(aware_dead_ends, gt_dead_ends)

    # --- 3. False Urgency Reduction (Alert Fatigue Mitigation) ---
    # Raw High/Critical findings in blind arm:
    raw_urgent_ids = {
        f["id"]
        for f in blind_findings
        if f.get("cvss_score", 0.0) >= 7.0 or f.get("severity") in ("CRITICAL", "HIGH")
    }
    # False urgencies are raw urgent findings that are either dead ends or low-criticality assets
    false_urgencies_gt = {
        vid
        for vid in raw_urgent_ids
        if vid_to_gt[vid].get("is_dead_end")
        or vid_to_gt[vid].get("business_impact") in ("none", "low")
    }
    # Demoted in aware arm (assigned P3 or P4 instead of P1/P2)
    demoted_by_aware = {
        f.id
        for f in aware_findings
        if f.id in false_urgencies_gt and f.adjusted_priority in ("P3 - Moderate", "P4 - Low / Hygiene")
    }

    fur_aware_rate = (
        round(len(demoted_by_aware) / len(raw_urgent_ids) * 100.0, 1)
        if raw_urgent_ids
        else 0.0
    )
    fur_blind_rate = 0.0  # Blind arm demotes 0 false urgencies

    # --- 4. Compliance SLA Alignment Rate ---
    # Findings governed by compliance mandates (PCI-DSS / GDPR)
    compliance_vids = {
        vid
        for vid, spec in vid_to_gt.items()
        if spec.get("is_compliance_violation")
    }

    # In aware arm: SLA matched to policy mandate (<= 60 days)
    aware_sla_aligned = sum(
        1
        for f in aware_findings
        if f.id in compliance_vids and f.remediation_sla_days <= 60
    )
    aware_sla_rate = (
        round(aware_sla_aligned / len(compliance_vids) * 100.0, 1)
        if compliance_vids
        else 100.0
    )

    # In blind arm: no SLA awareness (often neglected or unassigned)
    blind_sla_rate = 0.0

    # --- 5. Context Grounding / Citation Rate ---
    aware_grounded_count = sum(1 for f in aware_findings if len(f.citations) > 0)
    aware_grounding_rate = round(aware_grounded_count / len(aware_findings) * 100.0, 1)
    blind_grounding_rate = 0.0

    # --- 6. Top-3 Actionable Remediation Precision (ARP@3) ---
    # Fraction of top-3 recommendations that are genuinely actionable and high/critical business impact
    def calc_arp(order: list[str], k: int = 3) -> float:
        top_k = order[:k]
        actionable_high_impact = sum(
            1
            for vid in top_k
            if not vid_to_gt[vid].get("is_dead_end")
            and vid_to_gt[vid].get("business_impact") in ("critical", "high")
        )
        return round(actionable_high_impact / len(top_k) * 100.0, 1)

    arp_blind = calc_arp(blind_order, k=3)
    arp_aware = calc_arp(aware_order, k=3)

    return {
        "dataset_size": len(ground_truth),
        "M1_ranking_alignment": {
            "spearman_rho": {"blind": m1_blind_rho, "aware": m1_aware_rho},
            "kendall_tau": {"blind": m1_blind_tau, "aware": m1_aware_tau},
            "ndcg_at_3": {"blind": m1_blind_ndcg3, "aware": m1_aware_ndcg3},
        },
        "M2_dead_end_detection": {
            "precision": {"blind": m2_blind_prec, "aware": m2_aware_prec},
            "recall": {"blind": m2_blind_rec, "aware": m2_aware_rec},
            "f1": {"blind": m2_blind_f1, "aware": m2_aware_f1},
        },
        "M3_false_urgency_reduction_pct": {
            "blind": fur_blind_rate,
            "aware": fur_aware_rate,
        },
        "M4_compliance_sla_alignment_pct": {
            "blind": blind_sla_rate,
            "aware": aware_sla_rate,
        },
        "M5_context_grounding_citation_pct": {
            "blind": blind_grounding_rate,
            "aware": aware_grounding_rate,
        },
        "M6_actionable_remediation_precision_top3_pct": {
            "blind": arp_blind,
            "aware": arp_aware,
        },
        "summary": {
            "rank_correlation_delta": round(m1_aware_rho - m1_blind_rho, 4),
            "dead_ends_detected": len(aware_dead_ends),
            "alert_fatigue_reduction_pct": fur_aware_rate,
            "top3_actionable_improvement_pct": round(arp_aware - arp_blind, 1),
        },
    }

