#!/usr/bin/env python3
"""Demonstrate the Phase 1 retrieval interface with three example queries.

Usage:
    python scripts/demo_retrieval.py

Shows, for each query: the raw filters, the formatted per-chunk results
(distance + metadata), and the prompt-ready [ORG CONTEXT] block that a
downstream agent would inject into an LLM prompt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from org_rag_phase1.src.retrieve import (  # noqa: E402
    format_context_for_prompt,
    retrieve_org_context,
)

DEMO_QUERIES = [
    {
        "query": "What are the compliance requirements for payment systems?",
        "filters": None,
    },
    {
        "query": "Business criticality of asset 10.0.1.5",
        "filters": {"source_type": "asset"},
    },
    {
        "query": "Firewall restrictions on DMZ access",
        "filters": {"source_type": "topology"},
    },
]


def main() -> int:
    """Run the demo queries. Returns process exit code."""
    for i, demo in enumerate(DEMO_QUERIES, start=1):
        print("=" * 78)
        print(f"DEMO {i}: {demo['query']}")
        print(f"filters: {demo['filters']}")
        print("-" * 78)

        results = retrieve_org_context(demo["query"], filters=demo["filters"], top_k=3)
        if not results:
            print("No results — did you run scripts/build_index.py first?")
            return 1

        for rank, r in enumerate(results, start=1):
            print(
                f"#{rank} distance={r['distance']:.4f} "
                f"source_file={r['source_file']} type={r['source_type']} "
                f"criticality={r['asset_criticality']} "
                f"compliance={r['compliance_scope']} authority={r['authority_level']}"
            )
            text = r["chunk_text"].replace("\n", " ")
            print(f"   {text[:160]}{'...' if len(text) > 160 else ''}")

        print("-" * 78)
        print(format_context_for_prompt(results))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
