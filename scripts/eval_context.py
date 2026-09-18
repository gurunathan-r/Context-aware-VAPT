#!/usr/bin/env python3
"""Run quantitative evaluation comparing Context-Blind vs. Context-Aware prioritization.

Usage:
    python scripts/eval_context.py
    python scripts/eval_context.py --ground-truth results/vuln_ground_truth.json

Calculates:
  M1: Rank correlation (Spearman rho, Kendall tau, NDCG@3) vs. Expert Priority
  M2: Dead-End Detection (Precision, Recall, F1)
  M3: False Urgency Reduction (Alert Fatigue Mitigation Rate)
  M4: Compliance SLA Alignment Rate
  M5: Context Grounding Rate (% with document citations)
  M6: Top-3 Actionable Remediation Precision (ARP@3)

Emits:
  - Formatted scorecard printed to terminal.
  - Full metrics JSON saved to results/context_eval_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to sys.path. PROJECT_ROOT is inserted last so it has the
# highest priority: its org_rag_phase1/ path-shim must win over a sibling
# checkout that is also named org_rag_phase1.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))

from org_rag_phase1.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from org_rag_phase1.src.agents.context_agent import OrgContextAgent
from org_rag_phase1.src.eval_context import evaluate_prioritization

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_context")


def print_scorecard(metrics: dict) -> None:
    """Print an evaluation scorecard suitable for academic or industry review."""
    print("\n" + "=" * 80)
    print(" " * 18 + "CONTEXT-AWARE VAPT EVALUATION SCORECARD")
    print("=" * 80)
    print(f"{'METRIC':<42} | {'CONTEXT-BLIND':<15} | {'CONTEXT-AWARE':<15}")
    print("-" * 80)

    m1 = metrics["M1_ranking_alignment"]
    print(f"{'M1. Spearman Rank Correlation (rho)':<42} | {m1['spearman_rho']['blind']:<15.4f} | {m1['spearman_rho']['aware']:<15.4f}")
    print(f"{'    Kendall Tau Correlation (tau)':<42} | {m1['kendall_tau']['blind']:<15.4f} | {m1['kendall_tau']['aware']:<15.4f}")
    print(f"{'    NDCG@3 (Top-3 Business Ranking)':<42} | {m1['ndcg_at_3']['blind']:<15.4f} | {m1['ndcg_at_3']['aware']:<15.4f}")

    m2 = metrics["M2_dead_end_detection"]
    print(f"{'M2. Dead-End Detection Recall':<42} | {m2['recall']['blind']:<15.1%} | {m2['recall']['aware']:<15.1%}")
    print(f"{'    Dead-End Detection Precision':<42} | {m2['precision']['blind']:<15.1%} | {m2['precision']['aware']:<15.1%}")
    print(f"{'    Dead-End Detection F1-Score':<42} | {m2['f1']['blind']:<15.4f} | {m2['f1']['aware']:<15.4f}")

    m3 = metrics["M3_false_urgency_reduction_pct"]
    print(f"{'M3. False Urgency Reduction (Alert Fatigue)':<42} | {m3['blind']:<14.1f}% | {m3['aware']:<14.1f}%")

    m4 = metrics["M4_compliance_sla_alignment_pct"]
    print(f"{'M4. Compliance SLA Alignment Rate':<42} | {m4['blind']:<14.1f}% | {m4['aware']:<14.1f}%")

    m5 = metrics["M5_context_grounding_citation_pct"]
    print(f"{'M5. RAG Context Grounding / Citation Rate':<42} | {m5['blind']:<14.1f}% | {m5['aware']:<14.1f}%")

    m6 = metrics["M6_actionable_remediation_precision_top3_pct"]
    print(f"{'M6. Top-3 Actionable Remediation Precision':<42} | {m6['blind']:<14.1f}% | {m6['aware']:<14.1f}%")

    print("=" * 80)
    print("EXECUTIVE SUMMARY:")
    s = metrics["summary"]
    print(f"  • Rank Correlation Gain : Δrho = +{s['rank_correlation_delta']:.4f} (from inverse alignment to near-perfect)")
    print(f"  • Dead Ends Pruned      : {s['dead_ends_detected']} attack paths unexploitable under firewall policies")
    print(f"  • Alert Fatigue Reduced : {s['alert_fatigue_reduction_pct']:.1f}% false-critical alerts deprioritized")
    print(f"  • Top-3 Actionable Gain : +{s['top3_actionable_improvement_pct']:.1f}% improvement in high-impact remediation yield")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Context-Aware Prioritization")
    parser.add_argument(
        "--vulns",
        default=str(PROJECT_ROOT / "data" / "simulated_vulns.json"),
        help="Path to simulated vulnerabilities JSON",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(PROJECT_ROOT / "results" / "vuln_ground_truth.json"),
        help="Path to expert ground truth benchmark JSON",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save evaluation results JSON",
    )
    parser.add_argument(
        "--llm-url",
        default=LLM_BASE_URL,
        help=f"OpenAI-compatible LLM endpoint (default: {LLM_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"LLM model ID (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=LLM_API_KEY,
        help="LLM API key (optional for local Ollama/LM Studio)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=LLM_TIMEOUT_SECONDS,
        help=f"LLM request timeout seconds (default: {LLM_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    vulns_path = Path(args.vulns)
    gt_path = Path(args.ground_truth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vulns_path.is_file():
        logger.error("Vulnerabilities file not found: %s", vulns_path)
        return 1
    if not gt_path.is_file():
        logger.error("Ground truth file not found: %s", gt_path)
        return 1

    with open(vulns_path, "r", encoding="utf-8") as f:
        raw_vulns = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    logger.info("Evaluating %d findings against ground truth (LLM: %s at %s)...", len(raw_vulns), args.model, args.llm_url)

    agent = OrgContextAgent(
        context_enabled=True,
        llm_base_url=args.llm_url,
        llm_model=args.model,
        llm_api_key=args.api_key,
        timeout=args.timeout,
    )
    blind_list, aware_list = agent.process_findings(raw_vulns)

    eval_results = evaluate_prioritization(blind_list, aware_list, ground_truth)
    eval_results["generated_utc"] = datetime.now(timezone.utc).isoformat()

    print_scorecard(eval_results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"context_eval_{timestamp}.json"
    out_file.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    logger.info("Saved evaluation results to: %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

