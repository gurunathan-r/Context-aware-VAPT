#!/usr/bin/env python3
"""Ask the organizational knowledge base a question — full RAG in one command.

Usage:
    python scripts/rag_query.py "What are the compliance requirements for payment systems?"
    python scripts/rag_query.py "Who can reach the HR database?" --filter source_type=topology
    python scripts/rag_query.py "..." --model qwen3.8-27b-abliterated --show-prompt

Requires the index to be built (scripts/build_index.py) and a local
OpenAI-compatible LLM server (LM Studio: `lms server start`).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from org_rag_phase1.config import LLM_MODEL  # noqa: E402
from org_rag_phase1.src.generate import answer_with_context  # noqa: E402


def parse_filter(pairs: list[str] | None) -> dict | None:
    """Parse --filter key=value pairs into a dict (value kept as string)."""
    if not pairs:
        return None
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Bad --filter {pair!r}: expected key=value")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    """Run one RAG query and print the grounded answer with its sources."""
    ap = argparse.ArgumentParser(description="Query the org knowledge base (RAG).")
    ap.add_argument("query", help="Your question")
    ap.add_argument("--filter", action="append", metavar="KEY=VALUE",
                    help="Metadata filter, repeatable (e.g. source_type=asset)")
    ap.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve")
    ap.add_argument("--model", default=LLM_MODEL, help="LLM model id on the server")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Disable unfiltered fallback on empty filtered results")
    ap.add_argument("--show-prompt", action="store_true",
                    help="Also print the exact prompt sent to the LLM")
    args = ap.parse_args()

    kwargs = {}
    if args.base_url:
        kwargs["base_url"] = args.base_url

    try:
        result = answer_with_context(
            args.query,
            filters=parse_filter(args.filter),
            top_k=args.top_k or 3,
            model=args.model,
            use_fallback=not args.no_fallback,
            **kwargs,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"QUERY:  {result['query']}")
    print(f"MODEL:  {result['model']}   fallback: {result['used_fallback']}")
    print("-" * 78)
    for i, s in enumerate(result["sources"], 1):
        print(
            f"[{i}] {s['source_file']}  type={s['source_type']}  "
            f"crit={s['asset_criticality']}  comp={s['compliance_scope']}  "
            f"dist={s['distance']:.4f}"
        )
    print("-" * 78)
    print(result["context_block"])
    print("-" * 78)
    print("ANSWER:")
    print(result["answer"])
    print("=" * 78)

    if args.show_prompt:
        print("\n--- EXACT PROMPT SENT TO LLM ---")
        print(result["prompt"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
