"""Recon Agent — context-driven reconnaissance on top of the org RAG.

This is the first Phase 2 agent. It closes the agent loop:

    1. ASK the organizational RAG what it already knows about a target
       (retrieve_org_context with source_type=asset/topology/policy filters).
    2. PLAN probes from that context (which ports matter for this OS/role,
       which paths the topology says are plausible).
    3. SAFELY PROBE live targets — network-safe by design:
         - private-RFC1918 targets only (hard guard, override is explicit),
         - TCP connect() checks with short timeouts (no SYN stealth, no scans
           of arbitrary ranges, no payload injection),
         - optional DNS lookups, disabled by default.
    4. RECORD every finding as a structured, timestamped record.
    5. PUBLISH findings back into the organizational RAG: a markdown intel
       document is written to data/intel/ with a sidecar .meta.json
       (source_type="intel"), then indexed so the next agent iteration (or the
       next recon run) retrieves its own prior findings — organizational
       memory that compounds.

Safety/ethics: probing infrastructure you do not own is illegal in most
jurisdictions. This agent is built for the research testbed (the sample
inventory IPs) and lab environments. Guardrails are documented and deliberate.

Planned next (documented in README "To implement"):
    - service-version fingerprinting from banners,
    - topology-aware probe planning (firewall-rule reasoning via RAG),
    - differential recon (diff findings vs. last run),
    - exploitation-planning agent consuming recon intel.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from org_rag_phase1.config import INTEL_DIR, RECON_DEFAULT_TARGETS
from org_rag_phase1.src.index import index_chunks
from org_rag_phase1.src.retrieve import retrieve_org_context
from org_rag_phase1.src.scope import check_scope

logger = logging.getLogger(__name__)

# Ports probed by default, chosen for a VAPT testbed, not a broad scan.
DEFAULT_PORTS = (22, 80, 443, 3389, 8443)

# Private ranges (RFC1918 + loopback + documentation ranges). Recon targets
# must resolve into one of these unless explicitly overridden.
_PRIVATE_NETS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "127.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",
    )
)

# RAG queries used to build context per target — this is the agent "reading"
# the organizational knowledge base before acting.
_CONTEXT_QUERIES = (
    ("asset", "What is the role, owner, operating system and criticality of asset {target}?"),
    ("topology", "Network reachability and firewall rules for {target}?"),
    ("policy", "Security policy requirements that apply to {target}?"),
)


@dataclass
class Finding:
    """One structured recon observation about one target."""

    target: str
    category: str          # "port_open" | "port_closed" | "port_filtered" | "error" | "info"
    detail: str            # human-readable detail, e.g. "tcp/443 open"
    severity_hint: str     # "info" | "low" | "medium" | "high" (context-weighted later)
    timestamp: str         # ISO-8601 UTC
    extra: dict[str, Any] = field(default_factory=dict)

    def as_line(self) -> str:
        """Compact one-line rendering used in the intel markdown report."""
        extra = f" ({self.extra['note']})" if self.extra.get("note") else ""
        return f"- [{self.category}] {self.target}: {self.detail}{extra}"


def _is_private_ip(host: str) -> bool:
    """True if host is a literal private/documentation IP address."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in net for net in _PRIVATE_NETS)


def _resolve_host(host: str) -> str | None:
    """Resolve a hostname to an IP string, or None if unresolvable."""
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def check_target_allowed(target: str, *, allow_public: bool = False) -> str:
    """Guardrail: only allow lab/private targets unless explicitly overridden.

    Args:
        target: Hostname or IP literal.
        allow_public: Explicit override for public targets (off by default).

    Returns:
        The resolved IP address the probes will target.

    Raises:
        ValueError: If the target is public and not overridden, or unresolvable.
    """
    if _is_private_ip(target):
        return target
    ip = _resolve_host(target)
    if ip and _is_private_ip(ip):
        return ip
    if allow_public:
        logger.warning(
            "PUBLIC TARGET OVERRIDE enabled for %r — ensure you have written "
            "authorization before probing this host.", target,
        )
        return ip or target
    raise ValueError(
        f"Target {target!r} is not a private/lab address. This agent only "
        "probes RFC1918/documentation ranges by default; pass "
        "allow_public=True (CLI: --allow-public) only with written authorization."
    )


def probe_port(host: str, port: int, timeout: float = 1.5) -> Finding:
    """Single TCP connect() probe — the only network action this agent takes.

    Args:
        host: IP/hostname to probe.
        port: TCP port number.
        timeout: Connect timeout in seconds.

    Returns:
        A Finding with category port_open (connection succeeded) or
        port_filtered (refused/timeout/unreachable — indistinguishable here,
        which is fine for safe recon).
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return Finding(target=host, category="port_open", detail=f"tcp/{port} open",
                           severity_hint="info", timestamp=now)
    except OSError:
        return Finding(target=host, category="port_filtered", detail=f"tcp/{port} filtered/closed",
                       severity_hint="info", timestamp=now)


def _extract_role(context_results: list[dict[str, Any]], target: str, window: int = 250) -> str:
    """Extract the asset role line nearest to the target's mention in context.

    Asset inventories often chunk several assets into one chunk, so the first
    "Role:" line is not necessarily this target's role. We look for the
    "Role:" occurrence closest AFTER a mention of the target string.
    """
    best: str | None = None
    best_gap: int = 10**9
    for rec in context_results:
        text = str(rec.get("chunk_text", ""))
        start = 0
        while True:
            hit = text.find(target, start)
            if hit == -1:
                break
            role_hit = text.find("Role:", hit, hit + window)
            if role_hit != -1:
                gap = role_hit - hit
                if gap < best_gap:
                    best_gap = gap
                    best = text[role_hit + len("Role:"):].splitlines()[0].strip(" -")
            start = hit + len(target)
    return best or "unknown role"


def plan_probes(context_results: list[dict[str, Any]], ports: tuple[int, ...] = DEFAULT_PORTS) -> list[int]:
    """Choose probe ports from retrieved organizational context.

    Rule-based Phase 2 starter: the topology/policy chunks are searched for
    hints (HTTPS-only, tcp/443 mentions, DB references); matched ports are
    probed first. Falls back to DEFAULT_PORTS. An LLM-driven planner can
    replace this later without changing the agent contract.

    Args:
        context_results: Chunks retrieved from the RAG about the target.
        ports: Candidate ports.

    Returns:
        Ordered list of ports to probe (deduplicated).
    """
    text = "\n".join(str(r.get("chunk_text", "")) for r in context_results).lower()
    ordered: list[int] = []
    if "https" in text or "443" in text:
        ordered.append(443)
    if "ssh" in text or "22" in text:
        ordered.append(22)
    if "database" in text or "sql" in text:
        ordered.append(3389)
    for p in ports:
        if p not in ordered:
            ordered.append(p)
    return ordered


class ReconAgent:
    """RAG-driven recon agent that records and republishes its findings."""

    def __init__(
        self,
        targets: list[str] | None = None,
        intel_dir: str | Path = INTEL_DIR,
        *,
        allow_public: bool = False,
        probe_timeout: float = 1.5,
        context_enabled: bool = True,
    ) -> None:
        """Configure the agent.

        Args:
            targets: IPs/hostnames to recon (defaults to config.RECON_DEFAULT_TARGETS).
            intel_dir: Where intel markdown reports are written.
            allow_public: Explicit guardrail override for public targets.
            probe_timeout: TCP connect timeout in seconds.
            context_enabled: When False, ``gather_context`` returns empty context
                — the "context-blind" arm used by the ablation study (metrics.md
                R-metrics). Everything else is identical.
        """
        self.targets = targets or list(RECON_DEFAULT_TARGETS)
        self.intel_dir = Path(intel_dir)
        self.allow_public = allow_public
        self.probe_timeout = probe_timeout
        self.context_enabled = context_enabled

    # -- Step 1: READ from the organizational RAG ---------------------------
    def gather_context(self, target: str) -> dict[str, list[dict[str, Any]]]:
        """Retrieve everything the org RAG knows about one target.

        When ``context_enabled=False`` (context-blind arm), returns empty lists
        for every query kind so the agent plans purely from defaults — exactly
        the "context-blind" condition the ablation study needs.

        Args:
            target: IP or hostname.

        Returns:
            Mapping query-kind ("asset" | "topology" | "policy") to retrieved
            chunk records (retrieve_org_context format).
        """
        if not self.context_enabled:
            return {kind: [] for kind, _ in _CONTEXT_QUERIES}
        context: dict[str, list[dict[str, Any]]] = {}
        for kind, template in _CONTEXT_QUERIES:
            query = template.format(target=target)
            context[kind] = retrieve_org_context(query, top_k=3)
            logger.info("RAG context [%s] for %s: %d chunk(s)",
                        kind, target, len(context[kind]))
        return context

    # -- Steps 2+3: PLAN and PROBE ------------------------------------------
    def _scope_record(self, target: str, resolved: str) -> tuple[dict[str, Any], str | None]:
        """Check bug-bounty scope for ``target`` ahead of any probing.

        Returns ``(scope_verdict, detail_or_None)``. The gating contract:
            - ``no_policy``  → proceed (fall back to the RFC1918 guardrail),
            - ``in_scope``   → proceed and record the authorization as context,
            - ``out_of_scope`` → BLOCK: return an info finding, never probe.
        """
        try:
            verdict = check_scope(target)
        except ValueError as exc:
            logger.warning("Scope check failed for %s: %s", target, exc)
            return {"status": "no_policy", "records": []}, None

        sources = sorted({r.get("source_file") for r in verdict["records"] if r.get("source_file")})
        detail = "published bug-bounty scope: " + ", ".join(sources) if sources else None
        return verdict, detail

    def run_recon(self, target: str, ports: tuple[int, ...] | None = None) -> list[Finding]:
        """Plan probes from RAG context, then safely probe the target.

        Args:
            target: Private/lab IP or hostname.
            ports: Explicit ports override; otherwise planned from context.

        Returns:
            List of findings (scope record if any, context summary record, then
            per-port findings). If the target's bug-bounty scope policy marks it
            out of scope, NO probing happens — a single blocking record is
            returned instead.

        Raises:
            ValueError: If the target fails the private-address guardrail.
        """
        resolved = check_target_allowed(target, allow_public=self.allow_public)

        scope_verdict, scope_detail = self._scope_record(target, resolved)
        if scope_verdict["status"] == "out_of_scope":
            now = datetime.now(timezone.utc).isoformat()
            logger.warning("Target %s is OUT of bug-bounty scope — not probing.", target)
            return [
                Finding(
                    target=resolved,
                    category="info",
                    detail=scope_detail or f"target {target} is outside published bug-bounty scope",
                    severity_hint="high",
                    timestamp=now,
                    extra={
                        "scope_status": "out_of_scope",
                        "scope_sources": sorted(
                            {r.get("source_file") for r in scope_verdict["records"]}
                        ),
                        "probes_planned": 0,
                    },
                )
            ]

        context = self.gather_context(target)

        flat = [c for chunks in context.values() for c in chunks]
        role = _extract_role(flat, target)

        planned = tuple(plan_probes(flat, ports or DEFAULT_PORTS))
        logger.info("Probing %s (%s) on ports %s", resolved, role, planned)

        findings = [
            Finding(
                target=resolved,
                category="info",
                detail=f"recon context: {role}",
                severity_hint="info",
                timestamp=datetime.now(timezone.utc).isoformat(),
                extra={
                    "asset_criticality": next(
                        (c.get("asset_criticality") for c in flat
                         if c.get("source_type") == "asset"), "unknown"),
                    "business_unit": next(
                        (c.get("business_unit") for c in flat
                         if c.get("source_type") == "asset"), "unknown"),
                    "context_chunks": len(flat),
                },
            )
        ]
        if scope_verdict["status"] == "in_scope":
            findings[0].extra["scope_status"] = "in_scope"
            findings[0].extra["scope_sources"] = sorted(
                {r.get("source_file") for r in scope_verdict["records"]}
            )
            findings[0].detail += f" | in bug-bounty scope: {scope_detail or 'policy evidence'}"
        findings.extend(probe_port(resolved, p, timeout=self.probe_timeout) for p in planned)
        return findings

    # -- Steps 4+5: RECORD and PUBLISH back into the RAG --------------------
    def publish_findings(
        self,
        run_findings: dict[str, list[Finding]],
        *, reset_collection: bool = False, index: bool = True,
    ) -> dict[str, Any]:
        """Write the intel report, then ingest+index it into the org RAG.

        The report is written as <name>.md plus <name>.md.meta.json with
        source_type="intel" and compliance_scope="none" — exactly the format
        the Phase 1 ingestion pipeline already understands, so the findings
        become retrievable organizational knowledge for the next agent run.

        Args:
            run_findings: Mapping target -> findings from run_recon().
            reset_collection: Rebuild the whole collection (default: upsert).
            index: If False, only write the files (dry-run in tests).

        Returns:
            {"report_path", "meta_path", "chunks_indexed", "report_text"}
        """
        self.intel_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"recon_findings_{run_id}"
        report_path = self.intel_dir / f"{name}.md"
        meta_path = self.intel_dir / f"{name}.md.meta.json"

        lines = [
            f"# Recon Findings — {run_id}",
            "",
            f"Targets: {', '.join(run_findings)}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]
        for target, findings in run_findings.items():
            lines.append(f"## Target {target}")
            lines.extend(f.as_line() for f in findings)
            lines.append("")

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")

        meta = {
            "source_type": "intel",
            "business_unit": "unknown",
            "asset_criticality": "medium",
            "compliance_scope": "none",
            "authority_level": "informational",
            "source_file": report_path.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        chunks_indexed = 0
        if index:
            from org_rag_phase1.src.ingest import ingest_all
            chunks = ingest_all(self.intel_dir)
            chunks_indexed = index_chunks(chunks, reset=reset_collection)

        logger.info("Intel report published: %s (%d chunks in index)",
                    report_path, chunks_indexed)
        return {
            "report_path": str(report_path),
            "meta_path": str(meta_path),
            "chunks_indexed": chunks_indexed,
            "report_text": report_text,
        }

    # -- Full agent loop -----------------------------------------------------
    def run(self, *, index_findings: bool = True) -> dict[str, Any]:
        """Complete recon cycle over all targets: read → plan → probe → publish.

        Args:
            index_findings: Publish + index findings into the RAG when True.

        Returns:
            {"findings": {target: [Finding, ...]}, "publish": publish_findings() result}
        """
        all_findings: dict[str, list[Finding]] = {}
        for target in self.targets:
            try:
                all_findings[target] = self.run_recon(target)
            except ValueError as exc:
                logger.error("Skipping target %s: %s", target, exc)
                all_findings[target] = [
                    Finding(target=target, category="error", detail=str(exc),
                            severity_hint="low",
                            timestamp=datetime.now(timezone.utc).isoformat())
                ]

        publish = self.publish_findings(all_findings, index=index_findings) \
            if index_findings else {"report_path": None, "chunks_indexed": 0}
        return {"findings": all_findings, "publish": publish}


def findings_to_dicts(findings: list[Finding]) -> list[dict[str, Any]]:
    """Serialize findings for JSON output (CLI/API friendly)."""
    return [asdict(f) for f in findings]
