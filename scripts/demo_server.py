#!/usr/bin/env python3
"""Interactive demo UI for the Context-Aware VAPT framework — local, zero-dependency.

    python scripts/demo_server.py [--port 8765]

Serves a single-page UI (scripts/demo_ui.html) plus a small JSON API that runs
each demo stage against the real backend, returning both the human output and a
structured backend trace:

    GET  /                -> the UI page
    GET  /api/health      -> {llm, index, model, ...}
    GET  /api/rag?q=...   -> grounded answer + retrieved chunks trace
    POST /api/recon       -> recon run snapshot (both arms) + metrics
    GET  /api/audit       -> Evaluation Agent Q1–Q8 audit report
    GET  /api/context     -> context-aware prioritization (blind vs aware)

Everything runs locally: ChromaDB + CPU embeddings + the configured
OpenAI-compatible LLM endpoint (config.LLM_BASE_URL). No new dependencies —
stdlib http.server only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))  # org_rag_phase1/ path-shim → this copy

from org_rag_phase1.config import LLM_BASE_URL, LLM_MODEL, PROJECT_ROOT as ROOT  # noqa: E402
from org_rag_phase1.src.agents.eval_agent import ARM_AWARE, ReconEvalAgent, render_markdown  # noqa: E402
from org_rag_phase1.src.agents.recon import DEFAULT_PORTS, ReconAgent  # noqa: E402
from org_rag_phase1.src.eval import compute_metrics, snapshot_target  # noqa: E402
from org_rag_phase1.src.generate import answer_with_context  # noqa: E402
from org_rag_phase1.src.index import index_stats  # noqa: E402
from org_rag_phase1.src.retrieve import retrieve_org_context  # noqa: E402

UI_PATH = Path(__file__).resolve().parent / "demo_ui.html"
GT_PATH = ROOT / "results" / "ground_truth_testbed.json"


# ---------------------------------------------------------------------------
# Stage runners — each returns {"ok", "title", "human", "trace", "elapsed_ms"}
# ---------------------------------------------------------------------------

def _chunks_trace(records: list[dict]) -> list[dict]:
    """Compact chunk records for JSON transport / UI rendering."""
    out = []
    for r in records:
        out.append({
            "source_file": r.get("source_file"),
            "source_type": r.get("source_type"),
            "distance": round(float(r.get("distance", 0.0)), 4) if r.get("distance") is not None else None,
            "criticality": r.get("asset_criticality"),
            "compliance": r.get("compliance_scope"),
            "authority": r.get("authority_level"),
            "preview": str(r.get("chunk_text", ""))[:220],
        })
    return out


def _scrub_reasoning(text: str) -> str:
    """Strip <think> blocks AND untagged chain-of-thought preambles.

    Some local models emit reasoning without <think> tags ("Here's a thinking
    process: … 2. **Draft the answer:** …"). The answer contract requires the
    final response to start directly with the grounded content, so cut at the
    first structural marker (heading, bullet, bold section start) — the same
    convention rag_query.py output implies.
    """
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
    text = re.sub(r"^\s*<think>.*", "", text, flags=re.DOTALL)
    m = re.search(
        r"^(#{1,3} )|(^\s*[-*] )|(^\*\*)|(^(Final|Answer|Response)\b.*)$",
        text, flags=re.MULTILINE | re.IGNORECASE,
    )
    if m and m.start() > 0:
        text = text[m.start():]
    return text.strip()


def run_rag(query: str) -> dict:
    t0 = time.perf_counter()
    # Stage 1: raw retrieval trace (what the agent "reads").
    raw = retrieve_org_context(query, top_k=3)
    retrieval_trace = _chunks_trace(raw)
    # Stage 2: the full RAG loop (retrieve → prompt → local LLM → citations).
    result = answer_with_context(query)
    ms = int((time.perf_counter() - t0) * 1000)
    answer_raw = str(result.get("answer") or "")
    answer = _scrub_reasoning(answer_raw)
    reasoning_stripped = answer != answer_raw.strip()
    return {
        "ok": True,
        "title": f"RAG query — {query!r}",
        "human": {
            "answer": answer,
            "reasoning_stripped": reasoning_stripped,
            "sources": result.get("sources"),
            "used_fallback": result.get("used_fallback"),
            "model": result.get("model"),
            "prompt_chars": len(result.get("prompt") or ""),
            "context_block_chars": len(result.get("context_block") or ""),
        },
        "trace": {
            "steps": [
                {"step": "retrieve", "detail": f"embed query → cosine search → top {len(raw)} chunk(s)"},
                {"step": "augment", "detail": f"build grounded prompt ({len(result.get('prompt') or '')} chars) with citation contract"},
                {"step": "generate", "detail": f"local LLM {result.get('model')} → strip <think> → verify citations"},
            ],
            "retrieved_chunks": retrieval_trace,
        },
        "elapsed_ms": ms,
    }


def run_recon() -> dict:
    t0 = time.perf_counter()
    targets = ["10.0.1.5", "10.0.3.20"]
    gt_map = {}
    if GT_PATH.is_file():
        from org_rag_phase1.src.eval import load_ground_truth
        gt_map = load_ground_truth(GT_PATH)

    aware_agent = ReconAgent(targets=targets, context_enabled=True)
    blind_agent = ReconAgent(targets=targets, context_enabled=False)
    snapshots, per_target = [], []
    for target in targets:
        aware = snapshot_target(aware_agent, target, gt_map.get(target), probe=False)
        blind = snapshot_target(blind_agent, target, gt_map.get(target), probe=False)
        snapshots += [aware, blind]
        per_target.append({
            "target": target,
            "role": aware.get("role"),
            "criticality": aware.get("criticality"),
            "aware_planned": aware.get("planned"),
            "blind_planned": blind.get("planned"),
            "context_chunks": {k: len(v) for k, v in (aware.get("context") or {}).items()},
            "role_blind": blind.get("role"),
        })

    metrics = compute_metrics(snapshots, gt_map, numbered=False)
    ms = int((time.perf_counter() - t0) * 1000)

    human_lines = ["RECON AGENT — plan-only run (no probes fired)\n"]
    for t in per_target:
        human_lines.append(
            f"◆ {t['target']}  role={t['role']}  criticality={t['criticality']}\n"
            f"  aware plan : {t['aware_planned']}\n"
            f"  blind plan : {t['blind_planned']}\n"
            f"  context    : {t['context_chunks']}\n")
    human_lines.append(
        f"B2 plan recall : {metrics.get('B2 plan_recall')}\n"
        f"B4 context sensitivity : {metrics.get('B4 context_sensitivity')}\n"
        f"R2 recall delta (aware − blind) : {metrics.get('R2 recall_delta_aware')}")

    return {
        "ok": True,
        "title": "Recon Agent — READ → PLAN (context-aware vs context-blind)",
        "human": {"text": "\n".join(human_lines)},
        "trace": {
            "steps": [
                {"step": "READ", "detail": "gather_context(): RAG queries per target (asset / topology / policy)"},
                {"step": "PLAN", "detail": "plan_probes(): PORT_HINT_RULES matched against retrieved text"},
                {"step": "ablation", "detail": "blind arm re-runs with context_enabled=False (empty RAG)"},
            ],
            "per_target": per_target,
            "metrics": {k: v for k, v in metrics.items() if v is not None},
        },
        "elapsed_ms": ms,
    }


def run_audit() -> dict:
    t0 = time.perf_counter()
    targets = ["10.0.1.5", "10.0.3.20"]
    gt_map = {}
    if GT_PATH.is_file():
        from org_rag_phase1.src.eval import load_ground_truth
        gt_map = load_ground_truth(GT_PATH)

    aware_agent = ReconAgent(targets=targets, context_enabled=True)
    blind_agent = ReconAgent(targets=targets, context_enabled=False)
    snapshots = []
    for target in targets:
        snapshots.append(snapshot_target(aware_agent, target, gt_map.get(target), probe=False))
        snapshots.append(snapshot_target(blind_agent, target, gt_map.get(target), probe=False))

    evaluator = ReconEvalAgent()  # deterministic judge: offline, byte-stable
    report = evaluator.audit(snapshots, gt_map=gt_map)
    ms = int((time.perf_counter() - t0) * 1000)

    checks = [
        {"metric": c.metric, "name": c.name, "severity": c.severity,
         "passed": c.passed, "detail": c.detail[:220]}
        for c in report.checks if c.passed is not None
    ]
    return {
        "ok": True,
        "title": "Evaluation Agent — independent Q1–Q8 audit",
        "human": {
            "verdict": report.verdict,
            "grade": report.grade,
            "score": report.score,
            "metrics": report.metrics,
            "by_arm": report.by_arm,
            "markdown": render_markdown(report),
        },
        "trace": {
            "steps": [
                {"step": "collect", "detail": f"re-read {len(snapshots)} snapshot artifacts (never the agent's internals)"},
                {"step": "check", "detail": f"run {len(report.checks)} adversarial checks (critical/major/minor)"},
                {"step": "grade", "detail": "severity-weighted Q scores → composite → PASS/WARN/FAIL"},
            ],
            "checks": checks,
            "n_checks": len(report.checks),
            "n_failed": len(report.failures),
        },
        "elapsed_ms": ms,
    }


def run_context() -> dict:
    t0 = time.perf_counter()
    from org_rag_phase1.src.agents.context_agent import OrgContextAgent
    from org_rag_phase1.src.eval_context import evaluate_prioritization

    vulns_path = ROOT / "data" / "simulated_vulns.json"
    gt_path = ROOT / "results" / "vuln_ground_truth.json"
    if not vulns_path.is_file():
        return {"ok": False, "error": f"missing {vulns_path}"}

    raw_vulns = json.loads(vulns_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else {}

    agent = OrgContextAgent(context_enabled=True, use_llm=False)  # deterministic fallback analyzer
    blind_list, aware_list = agent.process_findings(raw_vulns)
    evaluation = evaluate_prioritization(blind_list, aware_list, ground_truth)
    ms = int((time.perf_counter() - t0) * 1000)

    def brief(items):
        rows = []
        for f in items:
            d = f if isinstance(f, dict) else f.to_dict()
            rows.append({
                "id": d.get("id"), "cve": d.get("cve"), "target": d.get("target"),
                "cvss": d.get("cvss_score") or d.get("base_cvss"),
                "risk": d.get("context_risk_score"),
                "priority": d.get("adjusted_priority"),
                "dead_end": d.get("is_dead_end"),
            })
        return rows

    m = evaluation.get("metrics", evaluation)
    m1 = m.get("M1_ranking_alignment", {}).get("spearman_rho", {})
    m2 = m.get("M2_dead_end_detection", {}).get("recall", {})
    m3 = m.get("M3_false_urgency_reduction_pct", {})
    m5 = m.get("M5_context_grounding_citation_pct", {})
    return {
        "ok": True,
        "title": "Context Agent — blind vs context-aware prioritization",
        "human": {
            "summary": (
                f"M1 rank correlation ρ: blind {m1.get('blind')} → aware {m1.get('aware')}"
                f"  (Δ {m.get('summary', {}).get('rank_correlation_delta')})\n"
                f"M2 dead-end recall: blind {m2.get('blind')} → aware {m2.get('aware')}\n"
                f"M3 alert-fatigue relief: {m3.get('aware')}%\n"
                f"M5 citation grounding: {m5.get('aware')}%"
            ),
            "note": ("UI runs the deterministic fallback analyzer for instant, "
                     "reproducible demos. With LLM reasoning (scripts/run_context_agent.py) "
                     "the aware arm reaches ρ = +1.00 (Δρ = +1.90)."),
        },
        "trace": {
            "steps": [
                {"step": "gather", "detail": "RAG context per finding (asset / topology / policy)"},
                {"step": "reason", "detail": "deterministic fallback analyzer (LLM offline in demo mode): dead-ends, criticality, SLAs"},
                {"step": "prioritize", "detail": "sort by context_risk_score; annotate citations"},
                {"step": "evaluate", "detail": "M1–M6 vs expert ground truth"},
            ],
            "blind_order": brief(blind_list),
            "aware_order": brief(aware_list),
            "metrics": {k: v for k, v in m.items() if isinstance(v, (int, float, str))},
        },
        "elapsed_ms": ms,
    }


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict | str, ctype: str = "application/json") -> None:
        body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, UI_PATH.read_text(encoding="utf-8"), "text/html")
            elif parsed.path == "/api/health":
                index = index_stats()
                llm_ok = False
                try:
                    import requests as _rq
                    llm_ok = _rq.get(f"{LLM_BASE_URL}/models", timeout=2).status_code == 200
                except Exception:
                    llm_ok = False
                self._send(200, {"index_chunks": index.get("total_chunks", 0),
                                 "llm": llm_ok, "llm_url": LLM_BASE_URL, "model": LLM_MODEL})
            elif parsed.path == "/api/rag":
                q = (parse_qs(parsed.query).get("q") or [""])[0].strip()
                if not q:
                    self._send(400, {"ok": False, "error": "missing ?q="})
                    return
                self._send(200, run_rag(q))
            elif parsed.path == "/api/audit":
                self._send(200, run_audit())
            elif parsed.path == "/api/context":
                self._send(200, run_context())
            else:
                self._send(404, {"ok": False, "error": "not found"})
        except Exception as exc:  # noqa: BLE001 — UI must always get JSON
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path == "/api/recon":
            try:
                self._send(200, run_recon())
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        sys.stderr.write("[demo] " + fmt % args + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive demo UI server (local, zero-dependency)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Demo UI:  {url}")
    print(f"Backend:  LLM={LLM_BASE_URL}  model={LLM_MODEL}")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
