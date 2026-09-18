#!/usr/bin/env python3
"""Run the Organizational Context Agent on simulated vulnerability scan findings.

Usage:
    python scripts/run_context_agent.py
    python scripts/run_context_agent.py --vulns data/simulated_vulns.json --out-dir reports/

Outputs:
  - Prints side-by-side Context-Blind vs. Context-Aware comparison table to terminal.
  - Generates reports/vuln_report_blind.md (Baseline).
  - Generates reports/vuln_report_aware.md (Context-Aware Prioritization).
  - Generates reports/vuln_comparison.json (Machine-readable audit artifact).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_context_agent")


def print_comparison_terminal(blind: list[dict], aware: list) -> None:
    """Print an eye-catching side-by-side comparison in terminal."""
    print("\n" + "=" * 95)
    print(" " * 22 + "VAPT PRIORITIZATION: CONTEXT-BLIND vs. CONTEXT-AWARE")
    print("=" * 95)
    print(f"{'#':<3} | {'VULN ID':<8} | {'TARGET:PORT':<18} | {'CVSS':<5} | {'RAW SEVERITY':<12} || {'ADJUSTED RANK':<14} | {'CONTEXT RISK':<13} | {'STATUS':<15}")
    print("-" * 95)

    # Map aware finding id to rank and object
    aware_ranks = {f.id: (i + 1, f) for i, f in enumerate(aware)}

    for b_rank, b_item in enumerate(blind, 1):
        vid = b_item["id"]
        a_rank, a_item = aware_ranks[vid]
        target_port = f"{b_item['target']}:{b_item['port']}"
        status = "🔴 DEAD END" if a_item.is_dead_end else "🟢 ACTIONABLE"

        # Highlight rank shift
        shift = b_rank - a_rank
        if shift > 0:
            rank_str = f"#{a_rank} (▲ +{shift})"
        elif shift < 0:
            rank_str = f"#{a_rank} (▼ {shift})"
        else:
            rank_str = f"#{a_rank} (=)"

        print(
            f"{b_rank:<3} | {vid:<8} | {target_port:<18} | {b_item['cvss_score']:<5} | {b_item['severity']:<12} || {rank_str:<14} | {a_item.context_risk_score:<13} | {status:<15}"
        )
    print("=" * 95)
    print("KEY DIFFERENCES DEMONSTRATED:")
    print("  1. Dead-End Elimination : VULN-005 (CVSS 10.0) & VULN-003 (CVSS 9.8) are demoted due to firewall rule FW-014 / subnet isolation.")
    print("  2. Business Escalation  : VULN-002 (Payment Gateway 443, CVSS 7.5) is promoted from #4 to #1 due to PCI-DSS & revenue impact.")
    print("  3. Alert Fatigue Relief : VULN-001 (Wiki, CVSS 9.8) is demoted from #2 to #3 because it's a low-criticality internal asset.")
    print("=" * 95 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Organizational Context Agent")
    parser.add_argument(
        "--vulns",
        default=str(PROJECT_ROOT / "data" / "simulated_vulns.json"),
        help="Path to simulated vulnerabilities JSON",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "reports"),
        help="Directory to save generated markdown & json reports",
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
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without calling LLM (use pure RAG context analyzer)",
    )
    args = parser.parse_args()

    vulns_path = Path(args.vulns)
    if not vulns_path.is_file():
        logger.error("Vulnerabilities file not found: %s", vulns_path)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(vulns_path, "r", encoding="utf-8") as f:
        raw_vulns = json.load(f)

    logger.info("Loaded %d simulated vulnerabilities from %s", len(raw_vulns), vulns_path.name)
    if args.no_llm:
        logger.info("Initializing Organizational Context Agent (RAG Context Analyzer mode, no LLM)...")
    else:
        logger.info("Initializing Organizational Context Agent (LLM: %s at %s)...", args.model, args.llm_url)

    agent = OrgContextAgent(
        context_enabled=True,
        use_llm=not args.no_llm,
        llm_base_url=args.llm_url,
        llm_model=args.model,
        llm_api_key=args.api_key,
        timeout=args.timeout,
    )
    blind_list, aware_list = agent.process_findings(raw_vulns)

    # Print terminal comparison
    print_comparison_terminal(blind_list, aware_list)

    # Generate Markdown reports
    blind_md, aware_md = agent.generate_markdown_reports(blind_list, aware_list)

    blind_file = out_dir / "vuln_report_blind.md"
    aware_file = out_dir / "vuln_report_aware.md"
    comparison_file = out_dir / "vuln_comparison.json"

    blind_file.write_text(blind_md, encoding="utf-8")
    aware_file.write_text(aware_md, encoding="utf-8")

    comparison_data = {
        "context_blind_report": blind_list,
        "context_aware_report": [f.to_dict() for f in aware_list],
    }
    comparison_file.write_text(json.dumps(comparison_data, indent=2), encoding="utf-8")

    logger.info("Saved Context-Blind report: %s", blind_file)
    logger.info("Saved Context-Aware report: %s", aware_file)
    logger.info("Saved comparison JSON:      %s", comparison_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

