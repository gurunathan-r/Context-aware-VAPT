#!/usr/bin/env python3
"""Run the Recon Agent: RAG-driven recon that republishes findings as intel.

Usage:
    python scripts/run_recon.py                                  # default testbed targets
    python scripts/run_recon.py --targets 10.0.1.5,10.0.3.20
    python scripts/run_recon.py --no-index                       # probe + report only (dry-run)
    python scripts/run_recon.py --fetch-scope example.com --targets app.example.com

The agent reads organizational context from the RAG, plans probes from that
context, safely probes private/lab targets (TCP connect, short timeout), and
writes its findings to data/intel/ as an intel document that is indexed back
into the same RAG — closing the loop (findings become retrievable knowledge).

Bug-bounty scope awareness: ``--fetch-scope DOMAIN`` downloads that program's
RFC 9116 security.txt policy and indexes it (source_type="scope") BEFORE the
run. The agent then checks each host against published scope and **never
probes an out-of-scope host**; in-scope and private hosts proceed normally.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from org_rag_phase1.src.agents.recon import (  # noqa: E402
    ReconAgent,
    findings_to_dicts,
)


def _fetch_and_index_scope(domains: list[str], timeout: float) -> int:
    """Fetch security.txt policies, index them, return total chunks."""
    from org_rag_phase1.src.scope import fetch_security_txt, index_scope_docs, save_scope_doc

    if not domains:
        return 0
    for domain in domains:
        payload = fetch_security_txt(domain, timeout=timeout)
        if payload is None:
            print(f"ERROR: no reachable security.txt policy for {domain} "
                  f"(RFC 9116 location not published).", file=sys.stderr)
            return -1
        save_scope_doc(domain, payload)
        print(f"Fetched scope policy for {domain} from {payload['url']} "
              f"(Contact: {payload['parsed'].get('Contact', 'not listed')})")
    total = index_scope_docs()
    print(f"Indexed {total} chunk(s) into the RAG (source_type=scope)")
    return total


def main() -> int:
    """Run recon and print findings + publication result."""
    ap = argparse.ArgumentParser(description="RAG-driven Recon Agent")
    ap.add_argument("--targets", default=None,
                    help="Comma-separated IPs/hostnames (default: config.RECON_DEFAULT_TARGETS)")
    ap.add_argument("--allow-public", action="store_true",
                    help="Explicitly allow public targets (requires written authorization)")
    ap.add_argument("--timeout", type=float, default=1.5, help="TCP connect timeout seconds")
    ap.add_argument("--no-index", action="store_true",
                    help="Do not index findings into the RAG (report only)")
    ap.add_argument("--fetch-scope", action="append", metavar="DOMAIN", default=None,
                    help="Fetch RFC 9116 security.txt scope policy for DOMAIN and index it "
                         "before the run (repeatable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.fetch_scope:
        indexed = _fetch_and_index_scope(args.fetch_scope, timeout=args.timeout)
        if indexed == -1:
            return 1

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    agent = ReconAgent(
        targets=targets,
        allow_public=args.allow_public,
        probe_timeout=args.timeout,
    )

    print("=" * 78)
    print("RECON AGENT — RAG-driven reconnaissance")
    print("=" * 78)
    result = agent.run(index_findings=not args.no_index)

    for target, findings in result["findings"].items():
        print(f"\n--- {target} ---")
        for f in findings:
            print(f"  {f.as_line()}")

    pub = result["publish"]
    if pub.get("report_path"):
        print("\n" + "-" * 78)
        print(f"Intel report:  {pub['report_path']}")
        print(f"Chunks in RAG: {pub['chunks_indexed']}")
        print("Next run will retrieve these findings as organizational memory.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
