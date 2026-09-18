#!/usr/bin/env python3
"""Run the recon evaluation harness (metrics.md) and emit a metrics report.

Usage:
    python scripts/eval_recon.py                                    # default testbed targets, offline
    python scripts/eval_recon.py --targets 10.0.1.5,10.0.2.10
    python scripts/eval_recon.py --ground-truth results/ground_truth_testbed.json
    python scripts/eval_recon.py --probe --allow-public --targets scanme.nmap.org
    python scripts/eval_recon.py --publish                      # include memory-loop (E) metrics

For every target the harness runs the SAME agent twice — context-aware vs
context-blind (``context_enabled=False``) — records each stage (READ/PLAN/
PROBE/RECORD) as a snapshot, and reduces snapshots + an optional lab profile
into A–E + R metrics. With ``--probe`` it performs real TCP-connect checks
(private RFC1918 targets only unless ``--allow-public``); without it, plans are
computed but no probes fire (fully offline, deterministic). Results JSON goes to
``results/recon_eval_<timestamp>.json``.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))  # org_rag_phase1/ path-shim → this copy

from org_rag_phase1.config import RECON_DEFAULT_TARGETS  # noqa: E402
from org_rag_phase1.src.agents.eval_agent import (  # noqa: E402
    ReconEvalAgent,
    render_markdown,
    report_text_from_snapshots,
)
from org_rag_phase1.src.eval import (  # noqa: E402
    ARM_AWARE,
    ARM_BLIND,
    compute_metrics,
    load_ground_truth,
    print_report,
    snapshot_target,
)
from org_rag_phase1.src.agents.recon import ReconAgent  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def _build_snapshots(
    agent_aware: ReconAgent,
    agent_blind: ReconAgent,
    targets: list[str],
    gt_map: dict,
    *,
    probe: bool,
    publish: bool,
) -> list[dict]:
    """Collect per-target snapshots for both arms, optionally publishing intel."""
    snapshots: list[dict] = []
    for target in targets:
        aware = snapshot_target(agent_aware, target,
                                gt_map.get(target), probe=probe)
        blind = snapshot_target(agent_blind, target,
                                gt_map.get(target), probe=probe)
        snapshots.append(aware)
        snapshots.append(blind)

        if publish:
            findings = {target: [f for f in aware["findings"]]}
            # Restore the context/scope records run_recon would emit so the
            # published intel report mirrors the agent's real artifact.
            try:
                pub = agent_aware.publish_findings(findings, index=True)
                aware["published"] = True
                aware["publish"] = pub
                aware["chunks_indexed"] = pub.get("chunks_indexed", 0)
                # E2/E3 retrievability: run the standard memory-loop query.
                from org_rag_phase1.src.retrieve import retrieve_org_context
                hits = retrieve_org_context(f"recon findings port scan results {target}", top_k=3)
                aware["intel_retrievable"] = any(
                    h["source_file"].startswith("recon_findings_") for h in hits
                )
                print(f"[eval] published intel for {target}: {pub['report_path']}")
            except Exception as exc:  # publish is best-effort for E-metrics
                logging.getLogger("eval_recon").warning("publish failed for %s: %s", target, exc)
                aware["published"] = False
    return snapshots


def main() -> int:
    """Run the evaluation and print the metrics report."""
    ap = argparse.ArgumentParser(description="Recon Agent evaluation metrics (metrics.md)")
    ap.add_argument("--targets", default=None,
                    help="Comma-separated targets (default: config.RECON_DEFAULT_TARGETS)")
    ap.add_argument("--ground-truth", default=None, metavar="FILE",
                    help="Lab profile JSON (role, expert_ports, listening_ports, ...)")
    ap.add_argument("--probe", action="store_true",
                    help="Actually run TCP-connect probes (off by default for determinism)")
    ap.add_argument("--allow-public", action="store_true",
                    help="Allow public targets when probing (written authorization required)")
    ap.add_argument("--timeout", type=float, default=1.5, help="TCP connect timeout seconds")
    ap.add_argument("--publish", action="store_true",
                    help="Publish aware-arm findings back into the RAG (enables E-metrics)")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="Write the full metrics JSON here (default: results/recon_eval_<ts>.json)")
    ap.add_argument("--numbered", action="store_true",
                    help="Key metrics by their metrics.md ids (A1, B2, R1, ...)")
    ap.add_argument("--audit", action="store_true",
                    help="Also run the Evaluation Agent (metrics.md §Q) and fill R5 "
                         "with its narrative judgement")
    ap.add_argument("--llm", action="store_true",
                    help="Let the Evaluation Agent use the local LLM as narrative "
                         "judge (falls back to the deterministic rubric)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else list(RECON_DEFAULT_TARGETS)
    gt_map = load_ground_truth(args.ground_truth) if args.ground_truth else {}

    aware_agent = ReconAgent(targets=targets, allow_public=args.allow_public,
                             probe_timeout=args.timeout, context_enabled=True)
    blind_agent = ReconAgent(targets=targets, allow_public=args.allow_public,
                             probe_timeout=args.timeout, context_enabled=False)

    snapshots = _build_snapshots(aware_agent, blind_agent, targets, gt_map,
                                 probe=args.probe, publish=args.publish)

    # Optional independent audit: it grades the same artifacts and supplies the
    # R5 narrative rating, which no metric in this module can compute itself.
    audit = None
    narrative_quality = None
    if args.audit:
        evaluator = ReconEvalAgent(use_llm=args.llm)
        narratives = {arm: report_text_from_snapshots(snapshots, arm)
                      for arm in (ARM_AWARE, ARM_BLIND)}
        audit = evaluator.audit(snapshots, gt_map=gt_map,
                                allow_public=args.allow_public, narratives=narratives)
        narrative_quality = audit.by_arm.get(ARM_AWARE, {}).get("Q7")
        print(render_markdown(audit, title="EVALUATION AGENT AUDIT (metrics.md §Q)"))

    report = compute_metrics(snapshots, gt_map, numbered=args.numbered,
                             narrative_quality=narrative_quality)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = args.out or RESULTS_DIR / f"recon_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "probe": args.probe,
        "ground_truth_file": args.ground_truth,
        "arms": [ARM_AWARE, ARM_BLIND],
        "metrics": report,
        "audit": audit.to_dict() if audit else None,
    }
    Path(out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print_report(report,
                 title=f"RECON EVAL — {', '.join(targets)} "
                       f"(mode={'live-probe' if args.probe else 'plan-only'})")
    print(f"\nResults JSON: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())