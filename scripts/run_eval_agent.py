#!/usr/bin/env python3
"""Audit the Recon Agent with the Evaluation Agent (metrics.md §Q).

Runs the recon agent over the testbed targets in both ablation arms (context-
aware / context-blind), captures the same snapshots `scripts/eval_recon.py`
measures, then independently audits those artifacts and grades the agent:

    python scripts/run_eval_agent.py                       # offline audit, both arms
    python scripts/run_eval_agent.py --ground-truth results/ground_truth_testbed.json
    python scripts/run_eval_agent.py --probe               # let the probes actually fire
    python scripts/run_eval_agent.py --publish             # include intel/memory-loop checks
    python scripts/run_eval_agent.py --llm                 # LLM narrative judge (R5/Q7)
    python scripts/run_eval_agent.py --snapshots results/recon_audit_<ts>.json   # re-audit a saved run
    python scripts/run_eval_agent.py --strict              # exit 1 when the verdict is FAIL

Writes `results/recon_audit_<timestamp>.json` (verdict, Q metrics, every check
with its evidence, the snapshots themselves) and a readable
`results/recon_audit_<timestamp>.md`. Exit code is 0 unless `--strict` is given
and the audit verdict is FAIL, so it can gate CI.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))  # org_rag_phase1/ path-shim → this copy

from org_rag_phase1.config import RECON_DEFAULT_TARGETS  # noqa: E402
from org_rag_phase1.src.agents.eval_agent import (  # noqa: E402
    ARM_AWARE,
    ARM_BLIND,
    ReconEvalAgent,
    render_markdown,
    report_text_from_snapshots,
    snapshots_to_dicts,
)
from org_rag_phase1.src.agents.recon import ReconAgent  # noqa: E402
from org_rag_phase1.src.eval import (  # noqa: E402
    load_ground_truth,
    print_report,
    snapshot_target,
)

RESULTS_DIR = PROJECT_ROOT / "results"


def _build_snapshots(
    targets: list[str],
    gt_map: dict,
    *,
    probe: bool,
    publish: bool,
    allow_public: bool,
    timeout: float,
) -> list[dict]:
    """Run both arms over every target and return their stage snapshots."""
    aware = ReconAgent(targets=targets, allow_public=allow_public,
                       probe_timeout=timeout, context_enabled=True)
    blind = ReconAgent(targets=targets, allow_public=allow_public,
                       probe_timeout=timeout, context_enabled=False)

    snapshots: list[dict] = []
    for target in targets:
        for agent in (aware, blind):
            snapshots.append(snapshot_target(agent, target, gt_map.get(target), probe=probe))

    if publish:
        for snapshot in [s for s in snapshots if s["arm"] == ARM_AWARE]:
            try:
                pub = aware.publish_findings(
                    {snapshot["target"]: snapshot["findings"]}, index=True,
                )
                snapshot["published"] = True
                snapshot["publish"] = pub
                snapshot["chunks_indexed"] = pub.get("chunks_indexed", 0)
            except Exception as exc:  # noqa: BLE001 — publish is best-effort for the audit
                logging.getLogger("run_eval_agent").warning(
                    "publish failed for %s: %s", snapshot["target"], exc)
                snapshot["published"] = False
    return snapshots


def _load_payload(path: str) -> tuple[list[dict], dict]:
    """Read snapshots (+ the run metadata) back from an audit's JSON payload.

    Returns ``(snapshots, payload)`` so a re-audit can reuse the recorded
    ground-truth file and therefore reproduce the original scores.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots")
    if not snapshots:
        raise ValueError(
            f"{path} contains no 'snapshots' key — re-run scripts/run_eval_agent.py "
            "to produce an auditable payload."
        )
    return snapshots, payload


def main() -> int:
    """Run the audit and print the report."""
    ap = argparse.ArgumentParser(
        description="Evaluation Agent — audit and grade the Recon Agent (metrics.md §Q)")
    ap.add_argument("--targets", default=None,
                    help="Comma-separated targets (default: config.RECON_DEFAULT_TARGETS)")
    ap.add_argument("--ground-truth", default=None, metavar="FILE",
                    help="Lab profile JSON enabling the Q6 fidelity checks")
    ap.add_argument("--snapshots", default=None, metavar="FILE",
                    help="Re-audit snapshots from a previous recon_audit_*.json "
                         "instead of running the recon agent")
    ap.add_argument("--probe", action="store_true",
                    help="Actually run TCP-connect probes (off by default for determinism)")
    ap.add_argument("--publish", action="store_true",
                    help="Publish aware-arm intel back into the RAG (enables Q4 sidecar checks)")
    ap.add_argument("--allow-public", action="store_true",
                    help="Acknowledge authorization to probe public targets")
    ap.add_argument("--timeout", type=float, default=1.5, help="TCP connect timeout seconds")
    ap.add_argument("--llm", action="store_true",
                    help="Use the local LLM as narrative judge for Q7 (metrics.md R5)")
    ap.add_argument("--llm-model", default=None, help="Override the judge model name")
    ap.add_argument("--llm-base-url", default=None,
                    help="Override the OpenAI-compatible judge endpoint")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="JSON payload path (default: results/recon_audit_<ts>.json)")
    ap.add_argument("--report", default=None, metavar="FILE",
                    help="Markdown audit report path (default: results/recon_audit_<ts>.md)")
    ap.add_argument("--no-report", action="store_true", help="Skip the markdown report")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero when the audit verdict is FAIL (CI gate)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    gt_map = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    targets = ([t.strip() for t in args.targets.split(",")] if args.targets
               else list(RECON_DEFAULT_TARGETS))

    if args.snapshots:
        snapshots, previous = _load_payload(args.snapshots)
        targets = sorted({str(s.get("target")) for s in snapshots})
        print(f"[audit] re-auditing {len(snapshots)} saved snapshot(s) from {args.snapshots}")
        # Reuse the recorded lab profile so the re-audit reproduces the scores.
        if not gt_map and previous.get("ground_truth_file"):
            recorded = Path(previous["ground_truth_file"])
            if recorded.is_file():
                gt_map = load_ground_truth(recorded)
                print(f"[audit] ground-truth profile restored from {recorded}")
    else:
        snapshots = _build_snapshots(targets, gt_map, probe=args.probe, publish=args.publish,
                                     allow_public=args.allow_public, timeout=args.timeout)

    # Narrative for Q7/R5: the published intel report when available, otherwise a
    # same-shaped plan-only rendering of that arm's findings.
    narratives: dict[str, str] = {}
    for arm in (ARM_AWARE, ARM_BLIND):
        published = next((s["publish"].get("report_text") for s in snapshots
                          if s.get("arm") == arm and s.get("publish", {}).get("report_text")),
                         None)
        narratives[arm] = published or report_text_from_snapshots(snapshots, arm)

    judge_kwargs = {}
    if args.llm_model:
        judge_kwargs["llm_model"] = args.llm_model
    if args.llm_base_url:
        judge_kwargs["llm_base_url"] = args.llm_base_url
    evaluator = ReconEvalAgent(use_llm=args.llm, **judge_kwargs)

    report = evaluator.audit(snapshots, gt_map=gt_map,
                             allow_public=args.allow_public, narratives=narratives)

    print_report(report.metrics, title="EVALUATION AGENT — Q METRIC SCORES (0-1)")
    print()
    for arm, scores in report.by_arm.items():
        rendered = ", ".join(f"{k}={'n/a' if v is None else f'{v:.2f}'}" for k, v in scores.items())
        print(f"{arm:>6} arm: {rendered}")
    print()
    print(f"VERDICT: {report.verdict}  |  grade {report.grade}  |  "
          f"quality score {report.score:.1f}/100")
    failures = report.graded_failures
    if failures:
        print(f"\n{len(failures)} failed check(s) in the {report.primary_arm} arm:")
        for check in failures:
            print(f"  - [{check.severity}] {check.metric}/{check.name}: {check.detail}")
    else:
        print(f"\nAll applicable checks passed in the {report.primary_arm} arm.")
    if report.control_failures:
        print(f"{len(report.control_failures)} expected control-arm difference(s) "
              "(context-blind baseline) — see the markdown report.")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"recon_audit_{stamp}.json"
    payload = {
        "generated_utc": datetime.now().astimezone().isoformat(),
        "targets": targets,
        "probe": args.probe,
        "publish": args.publish,
        "ground_truth_file": args.ground_truth,
        "judge": {"use_llm": args.llm, "model": args.llm_model or "config default"},
        "audit": report.to_dict(),
        "snapshots": snapshots_to_dicts(snapshots),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nAudit JSON: {out_path}")

    if not args.no_report:
        md_path = Path(args.report) if args.report else RESULTS_DIR / f"recon_audit_{stamp}.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Audit report: {md_path}")

    if args.strict and report.verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
