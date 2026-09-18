#!/usr/bin/env python3
"""Fetch a bug-bounty program's scope policy and index it into the RAG.

Usage:
    python scripts/fetch_scope.py example.com
    python scripts/fetch_scope.py example.com --no-index   # save only, no RAG

Deliberately standards-based and keyless: Fetches RFC 9116
``security.txt`` (https, then http; ``/.well-known/security.txt`` then
``/security.txt``) and saves it as ``data/scope/scope_<domain>.md`` with a
``source_type="scope"`` sidecar. With ``--index`` (default) it is ingested and
embedded into the same ChromaDB collection the Recon Agent reads, so subsequent
recon runs can tell in-scope from out-of-scope hosts before probing.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))  # org_rag_phase1/ path-shim → this copy

from org_rag_phase1.config import SECURITY_TXT_TIMEOUT  # noqa: E402


def main() -> int:
    """Fetch scope policy(ies) and optionally index them."""
    ap = argparse.ArgumentParser(description="Fetch + index bug-bounty scope policies (RFC 9116)")
    ap.add_argument("domains", nargs="+", metavar="DOMAIN",
                    help="Program/company domains, e.g. example.com")
    ap.add_argument("--timeout", type=float, default=SECURITY_TXT_TIMEOUT,
                    help="Per-request timeout seconds")
    ap.add_argument("--no-index", action="store_true",
                    help="Save the policy files but do not index them into the RAG")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from org_rag_phase1.src.scope import (  # noqa: E402
        fetch_security_txt,
        index_scope_docs,
        save_scope_doc,
    )

    saved = 0
    for domain in args.domains:
        payload = fetch_security_txt(domain, timeout=args.timeout)
        if payload is None:
            print(f"ERROR: no reachable security.txt policy for {domain}.", file=sys.stderr)
            return 1
        paths = save_scope_doc(domain, payload)
        print(f"Saved scope policy for {domain}:")
        print(f"  doc:   {paths['doc_path']}")
        print(f"  meta:  {paths['meta_path']}")
        print(f"  from:  {payload['url']}")
        print(f"  canonical: {payload['parsed'].get('Canonical', '-')}")
        print(f"  contact:   {payload['parsed'].get('Contact', '-')}")
        saved += 1

    if not args.no_index and saved:
        total = index_scope_docs()
        print(f"Indexed {total} chunk(s) into the RAG with source_type=scope.")

    return 0


if __name__ == "__main__":
    sys.exit(main())