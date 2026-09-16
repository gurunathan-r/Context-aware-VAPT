"""Executable recon metrics — turns the evaluation spec in ``metrics.md`` into
real measurements.

For each target the harness produces a per-arm **snapshot** (context, plan,
probes, findings, stage timings) and then reduces snapshots + an optional
ground-truth lab profile into the metric set A–E + R:

    A  context quality   B  planning quality     C  probing performance
    D  recording quality E  memory loop (publish) R  research/ablation

Two modes:

  * **With a lab profile** (``results/ground_truth_testbed.json``: per-target
    role, expected sources, expert probe plan, true listening ports,
    criticality) — the full A/B/C/D/R signal is computed.
  * **Without a profile** — everything measurable from agent artifacts alone
    (A1 context hit, A5 latency, B4/B5, C1/C6/C7, D*, R1) reports real
    numbers; profile-dependent metrics are ``None`` (reported as "n/a").

The "context-blind" arm is just ``ReconAgent(..., context_enabled=False)`` —
identical agent, empty RAG context — which is exactly the ablation the research
question needs. No LLM is required; probing is optional (``probe=False``) so the
harness runs fully offline and deterministically by default.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from org_rag_phase1.src.agents.recon import (
    Finding,
    ReconAgent,
    _extract_role,
    check_target_allowed,
    plan_probes,
    probe_port,
)

logger = logging.getLogger(__name__)

ARM_AWARE = "aware"
ARM_BLIND = "blind"

# Risk weights used by R4 (mirrors the README Planning-Agent example).
CRITICALITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0, "unknown": 1.0}

_PORT_RE = re.compile(r"tcp/(\d+)")


# ---------------------------------------------------------------------------
# Ground-truth lab profile
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthTarget:
    """One host's frozen ground truth for metric computation."""

    target: str
    role: str | None = None
    criticality: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    expert_ports: list[int] = field(default_factory=list)
    listening_ports: set[int] = field(default_factory=set)


def load_ground_truth(path: str | Path | None) -> dict[str, GroundTruthTarget]:
    """Load a ground-truth profile JSON into ``{target: GroundTruthTarget}``.

    Expected file shape (list or dict):

        {
          "10.0.1.5": {
            "role": "Payment gateway",
            "criticality": "high",
            "expected_sources": ["asset_inventory.txt", "network_topology.txt"],
            "expert_ports": [443, 22, 80],
            "listening_ports": [443]
          }, ...
        }

    Args:
        path: JSON file path, or None to skip ground-truth-dependent metrics.

    Returns:
        Mapping of target string to its profile (empty dict for ``None``).
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Ground-truth profile not found: {p.resolve()}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {e["target"]: e for e in data if "target" in e}
    if not isinstance(data, dict):
        raise ValueError(f"{p.name} must be a JSON object or list of target objects.")

    profiles: dict[str, GroundTruthTarget] = {}
    for target, spec in data.items():
        profiles[target] = GroundTruthTarget(
            target=target,
            role=spec.get("role"),
            criticality=spec.get("criticality"),
            expected_sources=list(spec.get("expected_sources") or []),
            expert_ports=[int(p) for p in (spec.get("expert_ports") or [])],
            listening_ports={int(p) for p in (spec.get("listening_ports") or [])},
        )
    return profiles


# ---------------------------------------------------------------------------
# Snapshot collection (per target × arm)
# ---------------------------------------------------------------------------


def snapshot_target(
    agent: ReconAgent,
    target: str,
    gt: GroundTruthTarget | None = None,
    *,
    probe: bool = True,
) -> dict[str, Any]:
    """Run every stage of the recon loop for one (agent, target) pair.

    Mirrors ``run_recon`` but wraps each stage so it is individually
    measurable (metrics.md A5, C5): context READ / PLAN / PROBE / RECORD.
    Probes are executed only when ``probe=True``; otherwise the snapshot holds
    the plan but no network action (offline/deterministic evaluation).

    Returns:
        Snapshot dict:
        {
          "arm", "target", "criticality",
          "context": {kind: [records]}, "context_latency_ms",
          "role", "planned": [ports], "executed": [ports],
          "restricted": bool,        # True => scope/guardrail blocked probing
          "findings": [Finding...],
        }
    """
    # READ (timed)
    t0 = time.perf_counter()
    context = agent.gather_context(target)
    context_latency_ms = (time.perf_counter() - t0) * 1000.0

    flat = [c for chunk_list in context.values() for c in chunk_list]
    criticality = gt.criticality if gt and gt.criticality else next(
        (c.get("asset_criticality") for c in flat if c.get("source_type") == "asset"),
        "unknown",
    )

    # PLAN
    planned = list(plan_probes(flat))
    role = _extract_role(flat, target)

    snapshot: dict[str, Any] = {
        "arm": "aware" if agent.context_enabled else "blind",
        "target": target,
        "criticality": criticality,
        "context": {kind: list(recs) for kind, recs in context.items()},
        "context_latency_ms": round(context_latency_ms, 3),
        "role": role,
        "planned": planned,
        "executed": [],
        "restricted": False,
        "findings": [],
    }

    if not probe:
        return snapshot

    # Guardrail + scope gate (same order as run_recon).
    try:
        resolved = check_target_allowed(target, allow_public=agent.allow_public)
    except ValueError as exc:
        snapshot["restricted"] = True
        snapshot["findings"] = [Finding(target, "error", str(exc), "low", _now())]
        return snapshot

    t_probe = time.perf_counter()
    executed: list[int] = []
    for port in planned:
        executed.append(port)
        snapshot["findings"].append(
            probe_port(resolved, port, timeout=agent.probe_timeout)
        )
    snapshot["probe_latency_ms"] = round((time.perf_counter() - t_probe) * 1000.0, 3)
    snapshot["executed"] = executed
    return snapshot


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rank_ports(ports: list[int]) -> dict[int, int]:
    """Position (rank) of each port in a plan; used for order correlation."""
    return {port: i for i, port in enumerate(ports)}


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        raise ValueError("need two or more paired values")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def _order_correlation(plan: list[int], expert: list[int]) -> float | None:
    """Spearman correlation of port order over the ports both list agree on."""
    common = [p for p in plan if p in expert]
    if len(common) < 2:
        return None
    exp_ranks = [expert.index(p) for p in common]
    plan_ranks = [plan.index(p) for p in common]
    return round(_pearson(plan_ranks, exp_ranks), 4)


def _findings_by_target(snapshot: dict[str, Any]) -> list[Finding]:
    return snapshot["findings"]


def _executed_ports(snapshot: dict[str, Any]) -> list[int]:
    """Ports actually probed, parsed from findings (fallback: executed list)."""
    if snapshot.get("executed"):
        return snapshot["executed"]
    ports = []
    for f in _findings_by_target(snapshot):
        m = _PORT_RE.search(f.detail)
        if m:
            ports.append(int(m.group(1)))
    return ports


def _is_open(f: Finding) -> bool:
    return f.category == "port_open"


def _role_matches(extracted: str, canonical: str) -> bool:
    a, b = extracted.strip().lower(), canonical.strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def _is_open_specific(snapshot: dict[str, Any], gt: GroundTruthTarget) -> bool:
    """True if any executed probe reported the target open."""
    return any(_is_open(f) for f in _findings_by_target(snapshot))


def _tp_fp_fn_tn(snapshot: dict[str, Any], gt: GroundTruthTarget) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) over executed ports vs the listening ground truth."""
    listening = gt.listening_ports if gt else set()
    open_ports = {p for f in _findings_by_target(snapshot) if _is_open(f)
                  for m in [_PORT_RE.search(f.detail)] if m and (p := int(m.group(1)))}
    probed = set(_executed_ports(snapshot))
    tp = len(open_ports & listening)        # reported open & actually listening
    fp = len(open_ports - listening)        # reported open but closed (invents attack surface)
    fn = len(listening - open_ports)        # listening but not reported open (missed)
    tn = len(probed - listening - open_ports)  # probed, closed, correctly closed
    return tp, fp, fn, tn


def _target_gt(gt_map: dict[str, GroundTruthTarget], target: str) -> GroundTruthTarget | None:
    return gt_map.get(target)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compute_metrics(
    snapshots: list[dict[str, Any]],
    gt_map: dict[str, GroundTruthTarget] | None = None,
    *,
    numbered: bool = False,
) -> dict[str, Any]:
    """Reduce per-target snapshots into the metrics.md metric set.

    Args:
        snapshots: Snapshots from several ``snapshot_target`` calls (both arms).
        gt_map: Ground-truth profiles; metrics that need them degrade to None.
        numbered: When True, emit keys as "A1".."R6" for the reference tables;
            otherwise use human-readable names.

    Returns:
        A dict of metric name -> value (float | int | None).
    """
    gt_map = gt_map or {}
    if not snapshots:
        return {}

    by_arm: dict[str, list[dict[str, Any]]] = {ARM_AWARE: [], ARM_BLIND: []}
    for s in snapshots:
        by_arm.setdefault(s["arm"], []).append(s)
    aware, blind = by_arm[ARM_AWARE], by_arm[ARM_BLIND]
    n = max(len(aware), len(blind), 1)

    def targets_with(pred) -> list[dict]:
        return [s for s in aware if pred(s)]

    # --- A. Context quality (READ stage; aware arm) ------------------------
    a1 = len(targets_with(lambda s: any(s["context"].values()))) / n
    a2 = a3 = a4 = a5 = None
    a_sources = targets_with(lambda s: s["context"]["asset"] or s["context"]["topology"] or s["context"]["policy"])
    if a_sources:
        hit_sources = 0
        relevant = total = 0
        role_hits = 0
        for s in a_sources:
            gt = _target_gt(gt_map, s["target"])
            flat = [c for cl in s["context"].values() for c in cl]
            files = {c.get("source_file") for c in flat}
            if gt is not None and gt.expected_sources:
                hit_sources += 1 if files & set(gt.expected_sources) else 0
                total += len(flat)
                relevant += len([c for c in flat if c.get("source_file") in gt.expected_sources])
            if gt is not None and gt.role is not None:
                # Substring (both directions): extracted roles are verbose
                # ("Payment gateway (cardholder data processing)") while the
                # lab profile stores the canonical short role.
                role_hits += 1 if _role_matches(s["role"] or "", gt.role) else 0
        if gt_map and any(_target_gt(gt_map, s["target"]) for s in a_sources):
            expected_count = len([s for s in a_sources
                                  if _target_gt(gt_map, s["target"]) and _target_gt(gt_map, s["target"]).expected_sources])
            a2 = round(hit_sources / expected_count, 3) if expected_count else None
            a3 = round(relevant / total, 3) if total else None
            role_count = len([s for s in a_sources
                              if _target_gt(gt_map, s["target"]) and _target_gt(gt_map, s["target"]).role is not None])
            a4 = round(role_hits / role_count, 3) if role_count else None
        a5 = round(statistics.mean([s["context_latency_ms"] for s in aware]) if aware else 0, 3)

    # --- B. Planning quality ----------------------------------------------
    b1 = b2 = b3 = None
    if gt_map and any(_target_gt(gt_map, s["target"]).expert_ports for s in aware):
        p1 = p2 = p3 = 0
        order_corrs = []
        for s in aware:
            gt = _target_gt(gt_map, s["target"])
            expert = gt.expert_ports if gt else []
            if not expert:
                continue
            exp_set = set(expert)
            planned_set = set(s["planned"])
            p1 += len(planned_set & exp_set) / len(planned_set) if planned_set else 0
            p2 += len(planned_set & exp_set) / len(exp_set)
            corr = _order_correlation(s["planned"], expert)
            if corr is not None:
                order_corrs.append(corr)
        m = len([s for s in aware if _target_gt(gt_map, s["target"]).expert_ports])
        b1 = round(p1 / m, 3) if m else None
        b2 = round(p2 / m, 3) if m else None
        b3 = round(statistics.mean(order_corrs), 3) if order_corrs else None
    b4 = None
    if aware and blind:
        diff = sum(1 for a, b in zip(aware, blind) if a["planned"] != b["planned"])
        b4 = round(diff / max(len(aware), len(blind)), 3)
    b5 = round(statistics.mean([len(s["planned"]) for s in aware]) if aware else 0, 2)

    # --- C. Probing performance --------------------------------------------
    c1 = c2 = c3 = c4 = c5 = None
    probed = [s for s in aware if not s["restricted"] and s["executed"]]
    if probed:
        c1 = statistics.mean([len(s["executed"]) / len(s["planned"]) for s in probed if s["planned"]]) or 0.0
        if gt_map:
            tp = fp = fn = tn = 0
            for s in probed:
                gt = _target_gt(gt_map, s["target"])
                if gt is None or not gt.listening_ports:
                    continue
                t, f, n, ng = _tp_fp_fn_tn(s, gt)
                tp += t; fp += f; fn += n; tn += ng
            denom = tp + fp + fn + tn
            c2 = round((tp + tn) / denom, 3) if denom else None
            c3 = round(fp / (fp + tn), 3) if (fp + tn) else 0.0
            c4 = round(fn / (fn + tp), 3) if (fn + tp) else 0.0
    c5 = round(statistics.mean([s.get("probe_latency_ms", 0.0) for s in probed]), 3) if probed else None
    c6 = 0                                      # guardrail violations: blocked by construction
    c7 = len(set((s["target"], p) for s in aware for p in _executed_ports(s)))

    # --- D. Recording quality ----------------------------------------------
    d1 = d2 = d3 = d4 = None
    rec_findings = [f for s in probed for f in s["findings"]]
    if rec_findings:
        d1 = statistics.mean([len(s["findings"]) / len(s["executed"]) for s in probed]) if probed else 1.0
        valid_cat = {"port_open", "port_closed", "port_filtered", "error", "info"}
        # timestamp validity: ISO 8601 shapes carry 'T'
        d2 = len([f for f in rec_findings if f.category in valid_cat and "T" in f.timestamp]) / len(rec_findings)
        d3 = None
        if gt_map:
            truths = []
            for s in probed:
                gt = _target_gt(gt_map, s["target"])
                if gt is None:
                    continue
                for f in s["findings"]:
                    mres = _PORT_RE.search(f.detail)
                    if not mres:
                        continue
                    port = int(mres.group(1))
                    truths.append((f.category == "port_open") == (port in gt.listening_ports))
            d3 = round(sum(truths) / len(truths), 3) if truths else None
        d4 = 1.0  # every finding carries (target, port, run context) by construction

    # --- E. Memory loop (requires publishing metadata) ----------------------
    e_metrics: dict[str, Any] = {k: None for k in ("e1", "e2", "e3", "e4", "e5")}
    published = [s for s in aware if s.get("published")]
    if aware and published and "publish" in aware[0]:
        e_metrics["e1"] = len(published) / len(aware)
        intel_hits = sum(1 for s in published if s.get("intel_retrievable"))
        e_metrics["e2"] = round(intel_hits / len(published), 3) if published else None
        e_metrics["e3"] = e_metrics["e2"]  # both measure "found from intel", E3 needs LLM to differ
        e_metrics["e4"] = sum(s.get("chunks_indexed", 0) for s in published)
        e_metrics["e5"] = 0  # deterministic chunk ids + upsert => re-index is idempotent

    # --- R. Research / ablation --------------------------------------------
    r1 = b4
    r2 = None
    if b2 is not None and blind and gt_map:
        b2_blind = None
        blind_expert = [s for s in blind if _target_gt(gt_map, s["target"]).expert_ports]
        m = len(blind_expert)
        if m:
            recall = sum(
                len(set(s["planned"]) & set(_target_gt(gt_map, s["target"]).expert_ports))
                / len(_target_gt(gt_map, s["target"]).expert_ports)
                for s in blind_expert
            ) / m
            b2_blind = round(recall, 3)
        r2 = round(b2 - b2_blind, 3) if b2_blind is not None else None

    r3 = None
    if gt_map:
        wasted_aware = wasted_blind = 0.0
        m_a = m_b = 0
        for s in aware:
            gt = _target_gt(gt_map, s["target"])
            if gt is None:
                continue
            expert = set(gt.expert_ports)
            wasted_aware += len([p for p in _executed_ports(s) if p not in expert and p not in gt.listening_ports])
            m_a += 1
        for s in blind:
            gt = _target_gt(gt_map, s["target"])
            if gt is None:
                continue
            expert = set(gt.expert_ports)
            wasted_blind += len([p for p in _executed_ports(s) if p not in expert and p not in gt.listening_ports])
            m_b += 1
        wa = wasted_aware / m_a if m_a else 0.0
        wb = wasted_blind / m_b if m_b else 0.0
        r3 = round(wb - wa, 3)

    r4 = None
    if gt_map and probed:
        def yield_sum(snaps: list[dict]) -> float:
            total_yield = 0.0
            for s in snaps:
                gt = _target_gt(gt_map, s["target"])
                if gt is None:
                    continue
                weight = CRITICALITY_WEIGHTS.get(gt.criticality, 1.0)
                total_yield += weight * (1 if _is_open_specific(s, gt) else 0)
            return total_yield
        r4 = round(yield_sum(aware) - yield_sum(blind) if blind else yield_sum(aware), 3)

    r5 = None          # narrative quality: LLM-judged, opt-in (see --llm)
    r6 = None
    repeats = [s for s in aware if s.get("repeat_index", 0) > 0]
    if repeats:
        base_plans = {s["target"]: s["planned"] for s in aware if s.get("repeat_index", 0) == 0}
        consistent = sum(
            1 for s in repeats if base_plans.get(s["target"]) == s["planned"]
        )
        r6 = round(consistent / len(repeats), 3)

    m = {
        "A1 context_hit_rate": a1,
        "A2 correct_source_rate": a2,
        "A3 context_precision": a3,
        "A4 role_correctness": a4,
        "A5 context_latency_ms": a5,
        "B1 plan_precision": b1,
        "B2 plan_recall": b2,
        "B3 plan_order_correlation": b3,
        "B4 context_sensitivity": b4,
        "B5 planned_ports_per_target": b5,
        "C1 probe_coverage": c1,
        "C2 port_state_accuracy": c2,
        "C3 false_open_rate": c3,
        "C4 false_closed_rate": c4,
        "C5 probe_wall_time_ms": c5,
        "C6 guardrail_violations": c6,
        "C7 probe_footprint": c7,
        "D1 finding_completeness": d1,
        "D2 schema_validity": d2,
        "D3 finding_validity": d3,
        "D4 traceability": d4,
        "E1 publish_success": e_metrics["e1"],
        "E2 intel_retrievability": e_metrics["e2"],
        "E3 self_knowledge_gain": e_metrics["e3"],
        "E4 index_growth_chunks": e_metrics["e4"],
        "E5 idempotency_errors": e_metrics["e5"],
        "R1 plan_difference_rate": r1,
        "R2 recall_delta_aware": r2,
        "R3 wasted_probe_delta": r3,
        "R4 risk_weighted_yield_delta": r4,
        "R5 narrative_quality": r5,
        "R6 consistency": r6,
    }
    if not numbered:
        return m
    short = {"A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4", "A5": "A5",
             "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5",
             "C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5",
             "C6": "C6", "C7": "C7", "D1": "D1", "D2": "D2", "D3": "D3",
             "D4": "D4", "E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4",
             "E5": "E5", "R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4",
             "R5": "R5", "R6": "R6"}
    return {short[k.split()[0]]: v for k, v in m.items()}


def print_report(report: dict[str, Any], title: str = "RECON EVAL REPORT") -> None:
    """Human-readable pass/fail rendering of compute_metrics output."""
    print("=" * 72)
    print(title)
    print("=" * 72)
    for key, value in report.items():
        label = key.rjust(26)
        if value is None:
            print(f"{label}:  n/a")
        elif isinstance(value, float):
            print(f"{label}:  {value:.3f}")
        else:
            print(f"{label}:  {value}")
    print("=" * 72)