"""Bug-bounty program scope awareness for the Recon Agent.

A legal VAPT program lives or dies on scope: probing a host that the target's
bug-bounty program does not cover is both a violation of the program's rules
and, in most jurisdictions, unauthorized access. This module gives the agent
three things:

    1. FETCH   — pull the program's pubel scope policy via RFC 9116
                 (``security.txt``), the open, keyless way to discover public
                 bug-bounty scope. ``HackerOne``/``Bugcrowd`` program pages
                 require auth or scraping; ``security.txt`` is the designed
                 machine-readable contract and needs no API key.
    2. SAVE    — persist the policy as a markdown document + ``.meta.json``
                 sidecar with ``source_type="scope"`` so it joins the RAG.
    3. CHECK   — given a host, determine in/out-of-scope from the indexed
                 policy docs. **Only ever restricts probing, never enables
                 it**: a host with no scope policy keeps today's behavior and
                 still must pass the RFC1918 guardrail / ``--allow-public``.

Everything is local after the one deliberate network call in ``fetch_security_txt``;
the matching logic is pure and fully testable offline.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from org_rag_phase1.config import SCOPE_DIR, SCOPE_USER_AGENT, SECURITY_TXT_TIMEOUT
from org_rag_phase1.src.retrieve import retrieve_org_context

logger = logging.getLogger(__name__)

# RFC 9116 locations, tried in order (https first, then http fallback).
SECURITY_TXT_PATHS = ("/.well-known/security.txt", "/security.txt")

# Matches "example.com", "*.example.com", "sub.example.com/none". Reused to
# extract candidate in-scope hostnames from policy text.
_DOMAIN_RE = re.compile(r"(?<!\w)(?:\*\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+", re.IGNORECASE)
# Matches dotted quads with optional CIDR suffix (10.0.1.5 or 10.0.0.0/8).
_IP_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")

# Source-file values returned by scope records; used to name the doc + tag
# findings with provenance.
_SCOPE_FIELDS = ("Canonical", "Contact", "Expires", "Hiring", "Encryption")


def _normalize_host(host: str) -> str:
    """Lowercase, strip an optional trailing port, return the bare host."""
    host = host.strip().rstrip(".").lower()
    if ":" in host and host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host


def host_is_ip(host: str) -> bool:
    """True if ``host`` is a literal IP address (v4 or v6)."""
    h = _normalize_host(host)
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _parse_fields(text: str) -> dict[str, str]:
    """Extract RFC 9116 key: value fields from a security.txt body."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields.setdefault(key.strip().title(), value.strip())
    return fields


def _fetch_url(url: str, timeout: float) -> str | None:
    """GET a URL; return the body on 200 with content, else None."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": SCOPE_USER_AGENT},
            allow_redirects=True,
        )
        if response.status_code == 200 and response.text.strip():
            return response.text
    except requests.RequestException:
        logger.debug("security.txt attempt failed for %s", url)
    return None


def fetch_security_txt(domain: str, timeout: float = SECURITY_TXT_TIMEOUT) -> dict[str, Any] | None:
    """Fetch the bug-bounty scope policy for ``domain`` via RFC 9116.

    Tries ``https://<domain>/.well-known/security.txt`` then ``/security.txt``,
    with an http fallback for each. This is the only network call in the
    scope module, and it is deliberately the keyless, standards-based route.

    Args:
        domain: Program or company domain, e.g. ``example.com``.
        timeout: Per-request timeout in seconds.

    Returns:
        ``{"domain", "url", "text", "parsed"}`` where ``parsed`` maps RFC 9116
        field names to values (Canonical / Contact / Expires / ...), or None if
        no reachable/parseable policy exists on the domain.
    """
    domain = _normalize_host(domain)
    if host_is_ip(domain):
        raise ValueError(f"security.txt scope is defined per domain, not per IP: {domain!r}")

    for path in SECURITY_TXT_PATHS:
        for scheme in ("https", "http"):
            url = f"{scheme}://{domain}{path}"
            text = _fetch_url(url, timeout)
            if text:
                return {
                    "domain": domain,
                    "url": url,
                    "text": text,
                    "parsed": _parse_fields(text),
                }
    logger.warning("No security.txt scope policy found for %s", domain)
    return None


def save_scope_doc(domain: str, payload: dict[str, Any], scope_dir: str | Path = SCOPE_DIR) -> dict[str, Any]:
    """Persist a fetched policy as a markdown doc + sidecar in ``scope_dir``.

    The document is written in the exact format the Phase 1 ingestion pipeline
    understands (``.md`` + ``<name>.md.meta.json``), with ``source_type="scope"``
    so it is retrievable via ``filters={"source_type": "scope"}``. The
    ``authority_level`` is ``mandatory`` because scope is a legal boundary: it
    constrains what agents may probe.

    Args:
        domain: The program domain (used for the file name).
        payload: ``fetch_security_txt`` result.
        scope_dir: Directory to write into (auto-created).

    Returns:
        ``{"doc_path", "meta_path"}``.
    """
    dir_path = Path(scope_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    name = f"scope_{domain}.md"

    fields = payload.get("parsed", {})
    lines = [f"# Bug Bounty Program Scope — {domain}", "", f"Source: {payload['url']}"]
    for key in _SCOPE_FIELDS:
        if fields.get(key):
            lines.append(f"{key}: {fields[key]}")
    lines.append("")
    lines.append("## Policy text (verbatim)")
    lines.append("")
    lines.append(payload.get("text", ""))

    meta = {
        "source_type": "scope",
        "business_unit": "unknown",
        "asset_criticality": "high",  # scope ignorance is the most expensive mistake
        "compliance_scope": "none",
        "authority_level": "mandatory",
        "source_file": name,
    }
    doc_path = dir_path / name
    meta_path = dir_path / f"{name}.meta.json"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Scope policy saved: %s", doc_path)
    return {"doc_path": str(doc_path), "meta_path": str(meta_path)}


def index_scope_docs(scope_dir: str | Path = SCOPE_DIR, reset: bool = False) -> int:
    """Ingest + index every scope doc in ``scope_dir`` into the RAG.

    Args:
        scope_dir: Directory holding scope documents (and sidecars).
        reset: Rebuild the collection first (like build_index.py --reset).

    Returns:
        Total chunk count in the collection after indexing.
    """
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    chunks = ingest_all(scope_dir)
    logger.info("Indexing %d scope chunk(s)", len(chunks))
    return index_chunks(chunks, reset=reset)


# ---------------------------------------------------------------------------
# Scope matching (pure, offline, unit-testable)
# ---------------------------------------------------------------------------

def parse_scope_entries(text: str) -> set[str]:
    """Extract candidate in-scope hostnames / CIDRs from policy text.

    Heuristic: every domain-like token (optionally wildcarded) and every dotted
    quad / CIDR in the policy body counts as a scope entry. Deliberately
    inclusive — false positives only matter if they *influence* a probe, and
    the verdict logic below errs toward blocking when a policy genuinely talks
    about a host.
    """
    entries: set[str] = set()
    for line in text.splitlines():
        low = line.strip().lower()
        if not low:
            continue
        # RFC 9116 field lines rarely carry hostnames of interest (expiry dates,
        # signatures, encryption keys) — skip them to avoid junk entries.
        if low[0].isalnum() and low.split(":", 1)[0].strip().lower() in (
            "expires", "signature", "encryption", "preferred-languages",
            "acknowledgments", "hiring", "policy",
        ):
            continue
        # Only consider lines that look like scope declarations or bare lists
        # (e.g. "*.example.com", ".example.com", "10.0.1.0/24"), plus any
        # "scope: x" lines. Generic prose is still scanned by the regexes.
        if not (low.startswith(("*", ".", "-", "scope", "in-scope", "in_scope"))):
            entries.update(m.group(0).lower() for m in _DOMAIN_RE.finditer(line))
            entries.update(m.group(0) for m in _IP_CIDR_RE.finditer(line))
            continue
        # Declaration-ish lines: capture the token after markers too.
        cleaned = low.lstrip("-* .:")
        entries.update(m.group(0).lower() for m in _DOMAIN_RE.finditer(cleaned))
        entries.update(m.group(0) for m in _IP_CIDR_RE.finditer(cleaned))
    return entries


def _domain_matches(host: str, entry: str) -> bool:
    """True if host equals a scope domain or is a subdomain of it.

    ``*.example.com`` entries treat the bare apex domain as in scope too.
    """
    entry = entry.lower().strip(".")
    host = host.lower().strip(".")
    if entry.startswith("*."):
        entry = entry[2:]
    if host == entry:
        return True
    return host.endswith("." + entry)


def _ip_matches(host: str, entry: str) -> bool:
    """Match ``host`` against an IP/CIDR entry."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if "/" in entry:
        try:
            return ip in ipaddress.ip_network(entry, strict=False)
        except ValueError:
            return False
    try:
        return ipaddress.ip_address(entry) == ip
    except ValueError:
        return False


def target_in_scope(host: str, scope_texts: list[str]) -> bool:
    """True if ``host`` is covered by at least one scope policy text.

    Args:
        host: IP literal or hostname to check.
        scope_texts: RAG chunk texts from ``source_type="scope"`` records.

    Returns:
        Whether the host is in scope. An empty ``scope_texts`` is treated as
        False — but callers should prefer ``check_scope`` which distinguishes
        "no policy at all" from "covered/not covered" so the agent can keep
        probing testbed targets that have no public program.
    """
    entries: set[str] = set()
    for text in scope_texts:
        entries |= parse_scope_entries(text)

    host = _normalize_host(host)
    is_ip = host_is_ip(host)
    for entry in entries:
        if is_ip:
            if _ip_matches(host, entry):
                return True
        elif _domain_matches(host, entry):
            return True
    return False


# ---------------------------------------------------------------------------
# Agent-facing verdict
# ---------------------------------------------------------------------------

def retrieve_scope_policy(host: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Pull indexed ``scope`` chunks relevant to ``host`` from the RAG."""
    return retrieve_org_context(
        f"bug bounty program scope and authorized targets for {host}",
        filters={"source_type": "scope"},
        top_k=top_k,
    )


def check_scope(host: str, top_k: int = 5) -> dict[str, Any]:
    """Determine whether a host falls inside an indexed bug-bounty program scope.

    Returns:
        One of:
          - ``{"status": "no_policy", "records": []}`` — no scope documents
            mention this host; the caller must fall back to the RFC1918 /
            ``--allow-public`` guardrail (unchanged behavior).
          - ``{"status": "in_scope", "records": [...]}`` — a published program
            authorizes probing; recorded as context.
          - ``{"status": "out_of_scope", "records": [...]}`` — a published
            program explicitly covers this host's family and the host is NOT in
            it; the caller MUST NOT probe.
    """
    host = _normalize_host(host)
    if host_is_ip(host):
        # IPs are matched via the CIDR entries of domain policies, so still try
        # to find a policy; but a bare IP with no policy is "unknown", not "out".
        records = retrieve_scope_policy(host, top_k)
        if not records or not any(r.get("chunk_text", "") for r in records):
            return {"status": "no_policy", "records": records}
        texts = [str(r.get("chunk_text", "")) for r in records]
        return {
            "status": "in_scope" if target_in_scope(host, texts) else "out_of_scope",
            "records": records,
        }

    records = retrieve_scope_policy(host, top_k)
    if not records:
        return {"status": "no_policy", "records": []}
    texts = [str(r.get("chunk_text", "")) for r in records]
    return {
        "status": "in_scope" if target_in_scope(host, texts) else "out_of_scope",
        "records": records,
    }