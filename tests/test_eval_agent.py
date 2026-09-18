"""Tests for src/agents/eval_agent.py — the Recon Evaluation Agent.

Fully offline and deterministic:

  * the auditor is exercised against hand-built synthetic snapshots (no RAG, no
    network, no LLM) so every check and every Q metric has an explicit expected
    outcome;
  * the end-to-end path uses the isolated temp Chroma fixture with
    ``probe=False``, i.e. no packets leave the machine;
  * the LLM judge is pointed at a closed local port to prove the deterministic
    heuristic fallback produces a usable grade instead of an exception.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from org_rag_phase1.src.agents.eval_agent import (
    ARM_AWARE,
    ARM_BLIND,
    AuditReport,
    ReconEvalAgent,
    _composite,
    _grade,
    _score_metrics,
    render_markdown,
    report_text_from_snapshots,
    snapshots_to_dicts,
)
from org_rag_phase1.src.agents.recon import DEFAULT_PORTS, Finding, ReconAgent
from org_rag_phase1.src.eval import GroundTruthTarget, load_ground_truth, snapshot_target

TARGET = "10.0.1.5"

# Context that justifies every claim the aware arm makes about TARGET: the role
# line sits next to the target mention (as _extract_role expects) and the
# topology hints at tcp/443 + tcp/22, which the planner should honour.
CONTEXT_TEXT = (
    "Asset inventory: 10.0.1.5 — Role: Payment gateway. Owner: finance. "
    "Criticality: high. Operating system: Ubuntu 22.04.\n"
    "Network topology: HTTPS (tcp/443) is published through the DMZ; "
    "SSH (tcp/22) is restricted to the admin jump host.\n"
)


def _port_finding(target: str, port: int, open_: bool = False) -> Finding:
    category = "port_open" if open_ else "port_filtered"
    detail = f"tcp/{port} open" if open_ else f"tcp/{port} filtered/closed"
    return Finding(target, category, detail, "info", "2026-09-18T00:00:00+00:00")


def _info_finding(target: str, *, role: str = "Payment gateway",
                  extra: dict | None = None) -> Finding:
    return Finding(
        target, "info", f"recon context: {role}", "info", "2026-09-18T00:00:00+00:00",
        extra if extra is not None else {
            "asset_criticality": "high",
            "business_unit": "finance",
            "context_chunks": 1,
            "sources": ["asset_inventory.txt"],
        },
    )


def _snap(
    arm: str = ARM_AWARE,
    target: str = TARGET,
    *,
    role: str = "Payment gateway",
    planned: tuple[int, ...] = (443, 22, 80, 3389, 8443),
    executed: tuple[int, ...] | None = None,
    findings: list | None = None,
    context_text: str | None = CONTEXT_TEXT,
    criticality: str = "high",
    restricted: bool = False,
    info_extra: dict | None = None,
    **extra,
) -> dict:
    """Hand-built snapshot with a controllable context and probe outcome."""
    context = {"asset": [], "topology": [], "policy": []}
    if context_text:
        context["asset"] = [{
            "chunk_text": context_text,
            "source_file": "asset_inventory.txt",
            "source_type": "asset",
            "asset_criticality": criticality,
            "business_unit": "finance",
        }]
    executed = list(planned) if executed is None else list(executed)
    if findings is None:
        findings = [_info_finding(target, role=role, extra=info_extra)]
        findings += [_port_finding(target, p) for p in executed]
    return {
        "arm": arm,
        "target": target,
        "criticality": criticality,
        "context": context,
        "context_latency_ms": 12.5,
        "role": role,
        "planned": list(planned),
        "executed": executed,
        "restricted": restricted,
        "findings": findings,
        "probe_latency_ms": 40.0,
        **extra,
    }


def _audit(snapshots: list[dict], **kwargs) -> AuditReport:
    kwargs.setdefault("narratives", None)
    return ReconEvalAgent().audit(snapshots, **kwargs)


def _failed_names(report: AuditReport) -> set[str]:
    return {c.name for c in report.failures}


GT = {
    TARGET: GroundTruthTarget(TARGET, role="Payment gateway", criticality="high",
                              expected_sources=["asset_inventory.txt"],
                              expert_ports=[443, 22, 80], listening_ports={443}),
}


class TestCleanRun:
    """A well-behaved aware run must pass every applicable check."""

    def test_clean_aware_run_is_graded_pass(self):
        snap = _snap()
        report = _audit([snap], gt_map=GT,
                        narratives={ARM_AWARE: report_text_from_snapshots([snap], ARM_AWARE)})

        assert report.verdict == "PASS"
        assert report.grade == "A"
        assert report.score >= 95.0           # Q7 carries the graded rating, not 1.0
        assert report.failures == []
        assert report.primary_arm == ARM_AWARE
        assert report.metrics["Q1"] == 1.0
        assert report.metrics["Q3"] == 1.0

    def test_plan_only_run_is_not_penalised_for_missing_findings(self):
        """No probes executed ⇒ schema/completeness checks are n/a, not failures."""
        snap = _snap(executed=(), findings=[])
        report = _audit([snap])

        assert report.metrics["Q4"] is None
        assert "finding_schema" not in _failed_names(report)
        assert "findings_per_probe" not in _failed_names(report)

    def test_unpublished_run_marks_sidecar_not_applicable(self):
        report = _audit([_snap(executed=(), findings=[])])
        assert any(c.metric == "Q4" and c.name == "intel_sidecar" and c.passed is None
                   for c in report.checks)


class TestSafetyIntegrity:
    """Scope and target guardrails are hard-zero rules."""

    def test_out_of_scope_target_that_was_still_probed_is_a_critical_failure(self):
        blocked = _snap(target="203.0.113.9", planned=(443,), executed=(443,),
                        findings=[
                            _info_finding("203.0.113.9", role="unknown role",
                                          extra={"scope_status": "out_of_scope",
                                                 "scope_sources": ["example.txt"]}),
                            _port_finding("203.0.113.9", 443),
                        ])
        report = _audit([blocked])
        assert report.verdict == "FAIL"          # critical beats the score
        assert any(c.name == "scope_gate" and c.severity == "critical"
                   for c in report.hard_failures)

    def test_respected_scope_gate_passes(self):
        blocked = _snap(target="203.0.113.9", planned=(443,), executed=(),
                        findings=[_info_finding("203.0.113.9", role="unknown role",
                                                extra={"scope_status": "out_of_scope"})])
        report = _audit([blocked])
        assert not report.hard_failures

    def test_public_target_probe_fails_without_authorisation_flag(self):
        public = _snap(target="8.8.8.8", planned=(443,), executed=(443,))
        assert any(c.name == "private_targets" for c in _audit([public]).hard_failures)

        allowed = _snap(target="8.8.8.8", planned=(443,), executed=(443,))
        assert not _audit([allowed], allow_public=True).hard_failures

    def test_unplanned_probe_is_flagged(self):
        drift = _snap(planned=(443, 22), executed=(443, 22, 3389),
                      findings=[_info_finding(TARGET)] +
                               [_port_finding(TARGET, p) for p in (443, 22, 3389)])
        assert "plan_execution_integrity" in _failed_names(_audit([drift]))

    def test_probe_budget_is_bounded(self):
        greedy = _snap(planned=DEFAULT_PORTS + (25, 53), executed=DEFAULT_PORTS + (25, 53))
        assert "probe_budget" in _failed_names(_audit([greedy]))

    def test_hint_driven_plan_may_exceed_default_count(self):
        """Regression: a legitimate hint-driven plan (db host → 1433 + 3389) is
        6 ports on a 5-port default set. Membership in the candidate universe,
        not a count bound, is what the budget must enforce."""
        wider = _snap(planned=(1433, 3389, 443, 22, 80, 8443),
                      executed=(1433, 3389, 443, 22, 80, 8443))
        assert "probe_budget" not in _failed_names(_audit([wider]))

    def test_auditor_rederives_hints_from_the_planner_contract(self):
        """Q2 must use the planner's own hint table — drift would audit against
        a contract the planner never agreed to."""
        from org_rag_phase1.src.agents.eval_agent import _HINT_RULES
        from org_rag_phase1.src.agents.recon import PORT_HINT_RULES
        assert _HINT_RULES is PORT_HINT_RULES


class TestGroundingAndContextUse:
    """Q1/Q2 — the research claim: claims must be grounded, context must matter."""

    def test_role_absent_from_context_is_flagged(self):
        fabricated = _snap(context_text="Unrelated policy text with no asset facts.")
        assert "role_supported" in _failed_names(_audit([fabricated]))

    def test_role_matching_lab_profile_passes(self):
        report = _audit([_snap()], gt_map=GT)
        assert not any(c.name == "role_matches_ground_truth" and c.failed for c in report.checks)

    def test_blind_arm_grounding_is_not_applicable(self):
        """Silence must not be scored as grounded — the control abstains, so Q1 is n/a."""
        blind = _snap(arm=ARM_BLIND, role="unknown role", context_text=None,
                      findings=[_info_finding(TARGET, role="unknown role",
                                              extra={"asset_criticality": "unknown",
                                                     "business_unit": "unknown"})])
        report = _audit([blind], gt_map=GT, primary_arm=ARM_BLIND)
        assert report.metrics["Q1"] is None
        assert {c.name for c in report.checks} & {"claims_cited", "role_matches_ground_truth"} == set()

    def test_plan_identical_to_defaults_is_flagged(self):
        default_order = _snap(planned=tuple(DEFAULT_PORTS))
        assert "plan_uses_context" in _failed_names(_audit([default_order]))

    def test_missing_hinted_port_is_flagged(self):
        """Context advertises tcp/443 but the plan omits it."""
        missed = _snap(planned=(22, 80, 3389, 8443))
        assert "plan_hint_consistency" in _failed_names(_audit([missed]))

    def test_blind_arm_must_use_the_default_plan(self):
        ok = _snap(arm=ARM_BLIND, role="unknown role", context_text=None,
                   planned=tuple(DEFAULT_PORTS),
                   findings=[_info_finding(TARGET, role="unknown role",
                                           extra={"asset_criticality": "unknown",
                                                  "business_unit": "unknown"})])
        assert "plan_uses_context" not in _failed_names(_audit([ok], primary_arm=ARM_BLIND))

    def test_claims_are_verified_against_retrieved_documents(self):
        """A claim the retrieval never made is a finding, not a pass."""
        inflated = _snap(info_extra={"asset_criticality": "high", "business_unit": "hr"})
        report = _audit([inflated])
        check = next(c for c in report.checks if c.name == "claims_evidence_backed")
        assert check.passed is False
        assert "business_unit" in check.detail          # no retrieved doc says "hr"
        assert report.metrics["Q1"] < 1.0

    def test_recorded_provenance_must_resolve_to_retrieved_documents(self):
        """Citing a document the RAG never returned is a fabricated citation."""
        fabricated = _snap(info_extra={"asset_criticality": "high", "business_unit": "finance",
                                       "sources": ["invented_policy.txt"]})
        report = _audit([fabricated])
        check = next(c for c in report.checks if c.name == "claims_evidence_backed")
        assert check.passed is False
        assert "invented_policy.txt" in check.detail

    def test_run_level_role_claim_shared_across_targets_is_flagged(self):
        """The A4 regression class, caught without any ground-truth profile."""
        leaky = [
            _snap(target="10.0.1.5", role="Payment gateway"),
            _snap(target="10.0.2.10", role="Payment gateway",
                  context_text=CONTEXT_TEXT.replace("10.0.1.5", "10.0.2.10")),
        ]
        report = _audit(leaky)
        check = next(c for c in report.checks if c.name == "role_attribution_unique")
        assert check.passed is False

    def test_run_level_role_check_passes_for_distinct_roles(self):
        distinct = [
            _snap(target="10.0.1.5", role="Payment gateway"),
            _snap(target="10.0.2.10", role="Internal wiki",
                  context_text="Asset 10.0.2.10 Role: Internal wiki, internal only."),
        ]
        report = _audit(distinct)
        assert not any(c.name == "role_attribution_unique" for c in report.failures)


class TestHonestyCalibration:
    """Q5 — the blind control must abstain instead of inventing organisational facts."""

    def test_blind_arm_inventing_facts_fails(self):
        invented = _snap(arm=ARM_BLIND, role="unknown role", context_text=None,
                         findings=[_info_finding(TARGET, role="unknown role",
                                                 extra={"asset_criticality": "high",
                                                        "business_unit": "finance"})])
        report = _audit([invented], primary_arm=ARM_BLIND)
        assert "blind_arm_abstains" in _failed_names(report)

    def test_blind_arm_abstaining_passes(self):
        abstained = _snap(arm=ARM_BLIND, role="unknown role", context_text=None,
                          findings=[_info_finding(TARGET, role="unknown role",
                                                  extra={"asset_criticality": "unknown",
                                                         "business_unit": "unknown"})])
        report = _audit([abstained], primary_arm=ARM_BLIND)
        assert "blind_arm_abstains" not in _failed_names(report)


class TestGroundTruthFidelity:
    """Q6 — plan vs the frozen lab profile."""

    def test_missing_expert_port_lowers_recall(self):
        gt = {TARGET: GroundTruthTarget(TARGET, role="Payment gateway", criticality="high",
                                        expert_ports=[443, 1433], listening_ports={443})}
        report = _audit([_snap()], gt_map=gt)          # plan has no 1433
        check = next(c for c in report.checks
                     if c.name == "expert_recall" and c.metric == "Q6")
        assert check.passed is False
        assert "0.50" in check.detail

    def test_reported_open_port_that_is_closed_is_flagged(self):
        gt = {TARGET: GroundTruthTarget(TARGET, expert_ports=[443], listening_ports={443})}
        snap = _snap(planned=(443, 22), executed=(443, 22),
                     findings=[_info_finding(TARGET),
                               _port_finding(TARGET, 443, open_=True),
                               _port_finding(TARGET, 22, open_=True)])
        report = _audit([snap], gt_map=gt)
        assert any(c.name == "no_false_open" and c.failed for c in report.checks)

    def test_checks_degrade_without_a_profile(self):
        report = _audit([_snap()])
        assert report.metrics["Q6"] is None
        assert any(c.metric == "Q6" and c.passed is None for c in report.checks)

    def test_real_testbed_profile_loads_and_audits(self):
        from pathlib import Path
        from org_rag_phase1.config import PROJECT_ROOT
        gt_path = Path(__file__).resolve().parent.parent / "results" / "ground_truth_testbed.json"
        assert gt_path.is_file(), f"starter profile missing: {gt_path}"
        profiles = load_ground_truth(gt_path)
        assert profiles, "starter ground-truth profile should ship with the repo"
        report = _audit([_snap()], gt_map=profiles)
        assert report.metrics["Q6"] is not None


class TestNarrativeJudge:
    """Q7/R5 — LLM judge with a deterministic fallback."""

    def test_heuristic_judge_scores_a_structured_report(self):
        text = report_text_from_snapshots([_snap(executed=(), findings=[])], ARM_AWARE)
        judged = ReconEvalAgent().judge_narrative(text)
        assert judged["judge"] == "heuristic"
        assert judged["overall"] >= 6.0
        assert 0.0 <= judged["overall"] <= 10.0

    def test_unreachable_llm_falls_back_instead_of_raising(self):
        judge = ReconEvalAgent(use_llm=True, llm_base_url="http://127.0.0.1:9/v1",
                               llm_model="nonexistent", timeout=0.5)
        judged = judge.judge_narrative("# Report\n- [info] 10.0.1.5: nothing\n")
        assert judged["judge"] == "heuristic"

    def test_thin_narrative_fails_the_q7_check(self):
        report = _audit([_snap()], narratives={ARM_AWARE: "no header, no targets, no findings"})
        q7 = [c for c in report.checks if c.metric == "Q7"]
        assert q7 and any(c.failed for c in q7)

    def test_q7_metric_carries_the_graded_rating(self):
        """Q7 must be the graded 0–1 rating (R5), not a coarse pass/fail."""
        narrative = report_text_from_snapshots([_snap(executed=(), findings=[])], ARM_AWARE)
        report = _audit([_snap()], narratives={ARM_AWARE: narrative})
        judged = ReconEvalAgent().judge_narrative(narrative)
        assert report.by_arm[ARM_AWARE]["Q7"] == pytest.approx(judged["overall"] / 10.0, abs=0.001)

    def test_narrative_judgement_is_recorded_under_the_right_arm(self):
        report = _audit([_snap()], narratives={ARM_AWARE: report_text_from_snapshots([_snap()], "aware")})
        assert any(c.metric == "Q7" and c.detail.startswith(f"[{ARM_AWARE}:report]")
                   for c in report.checks)


class TestScoringMath:
    """Severity weighting, n/a renormalisation and grade mapping."""

    def test_severity_weights_a_failed_critical_harder_than_a_minor(self):
        from org_rag_phase1.src.agents.eval_agent import AuditCheck

        only_minor = _score_metrics([AuditCheck("Q3", "a", "minor", False, "x"),
                                     AuditCheck("Q3", "b", "critical", True, "y")])
        assert only_minor["Q3"] == pytest.approx(3 / 4)

        only_critical = _score_metrics([AuditCheck("Q3", "a", "major", False, "x"),
                                        AuditCheck("Q3", "b", "critical", True, "y")])
        assert only_critical["Q3"] == pytest.approx(3 / 5)

    def test_not_applicable_metrics_are_excluded_not_zeroed(self):
        metrics = _score_metrics([])
        assert all(v is None for v in metrics.values())
        assert _composite({"Q1": 1.0, "Q6": None}) == 100.0

    def test_composite_is_weighted_across_available_metrics(self):
        assert _composite({k: 1.0 for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")}) == 100.0
        assert _composite({k: 0.0 for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")}) == 0.0
        assert _composite({}) == 0.0

    @pytest.mark.parametrize("score,letter", [(95, "A"), (85, "B"), (72, "C"), (61, "D"), (10, "F")])
    def test_grade_mapping(self, score, letter):
        assert _grade(score) == letter


class TestSerialization:
    """The audit must survive a JSON round-trip (results/recon_audit_*.json)."""

    def test_report_to_dict_is_json_shaped(self):
        import json

        report = _audit([_snap()])
        payload = report.to_dict()
        json.dumps(payload)                       # must not raise
        assert payload["verdict"] in {"PASS", "WARN", "FAIL"}
        assert payload["primary_arm"] == ARM_AWARE
        assert set(payload["metrics"]) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"}
        assert all(set(c) == {"metric", "name", "severity", "passed", "score",
                              "detail", "evidence"} for c in payload["checks"])

    def test_dict_findings_are_auditable(self):
        snap = _snap()
        as_dicts = snapshots_to_dicts([snap])
        assert isinstance(as_dicts[0]["findings"][0], dict)
        report = _audit(as_dicts, gt_map=GT)
        assert report.verdict == "PASS"
        assert report.failures == []

    def test_markdown_report_lists_verdict_and_failures(self):
        broken = _snap(context_text="No asset facts here at all.")
        md = render_markdown(_audit([broken]))
        assert "Verdict:" in md
        assert "Q1" in md and "Failed checks" in md
        assert "role_supported" in md

    def test_control_arm_failures_are_reported_apart_from_the_verdict(self):
        snapshots = [_snap(), _snap(arm=ARM_BLIND, role="unknown role", context_text=None,
                                    findings=[_info_finding(TARGET, role="unknown role",
                                                            extra={"asset_criticality": "unknown",
                                                                   "business_unit": "unknown"})])]
        report = _audit(snapshots)
        assert {c.name for c in report.graded_failures} & {"context_consumed"} == set()
        assert any(c.name == "context_consumed" for c in report.control_failures)
        assert "Control-arm differences" in render_markdown(report)


@pytest.fixture(scope="module")
def indexed_audit(tmp_path_factory, embedder):
    """Isolated temp Chroma index over data/raw (same pattern as test_eval.py)."""
    from org_rag_phase1.config import RAW_DIR
    from org_rag_phase1.src import index as index_mod
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    root = tmp_path_factory.mktemp("audit")
    index_mod.CHROMA_PATH = str(root / "chroma_db")
    index_mod.get_embedder = lambda: embedder
    index_mod._client = None
    index_mod._collection = None
    index_chunks(ingest_all(RAW_DIR), reset=True)
    yield SimpleNamespace()
    index_mod._client = None
    index_mod._collection = None


class TestEndToEnd:
    """Audit real snapshots from the real agents — offline, no probes."""

    def test_audits_both_arms_from_real_agents(self, indexed_audit):
        aware_agent = ReconAgent(targets=[TARGET], context_enabled=True)
        blind_agent = ReconAgent(targets=[TARGET], context_enabled=False)
        snapshots = [
            snapshot_target(aware_agent, TARGET, probe=False),
            snapshot_target(blind_agent, TARGET, probe=False),
        ]
        evaluator = ReconEvalAgent()
        report = evaluator.audit(
            snapshots,
            narratives={arm: report_text_from_snapshots(snapshots, arm)
                        for arm in (ARM_AWARE, ARM_BLIND)},
        )

        assert report.verdict in {"PASS", "WARN", "FAIL"}
        assert set(report.by_arm) == {ARM_AWARE, ARM_BLIND}
        assert report.metrics["Q3"] == 1.0          # guardrails held, no probes fired
        assert not report.hard_failures
        assert len(report.targets) == 1

    def test_audit_reproduces_the_ablation_signal(self, indexed_audit):
        """The control arm must score strictly lower: it consumes no knowledge (Q8)."""
        aware_agent = ReconAgent(targets=[TARGET], context_enabled=True)
        blind_agent = ReconAgent(targets=[TARGET], context_enabled=False)
        snapshots = [
            snapshot_target(aware_agent, TARGET, probe=False),
            snapshot_target(blind_agent, TARGET, probe=False),
        ]
        report = ReconEvalAgent().audit(snapshots)
        aware, blind = report.by_arm[ARM_AWARE], report.by_arm[ARM_BLIND]

        assert aware["Q8"] == 1.0
        assert blind["Q8"] == 0.0
        assert _composite(aware) > _composite(blind)
        assert report.primary_arm == ARM_AWARE         # only the aware arm is graded
        assert report.metrics == aware
