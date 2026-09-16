#!/usr/bin/env python3
"""One-command build: ingest + chunk + embed + persist all org documents.

Usage:
    python scripts/build_index.py [--reset]

    --reset  Delete and recreate the collection first (fresh build).
             Without it, chunks are upserted idempotently.

Exit codes: 0 on success, 1 on any failure.
"""

import logging
import sys
from pathlib import Path

# Make the org_rag_phase1 package importable regardless of CWD: the package
# lives one level above the project root (repo root holds org_rag_phase1/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from org_rag_phase1.config import RAW_DIR  # noqa: E402
from org_rag_phase1.src.index import index_chunks, index_stats  # noqa: E402
from org_rag_phase1.src.ingest import ingest_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    """Build the vector index. Returns process exit code (0 ok, 1 failure)."""
    reset = "--reset" in sys.argv
    try:
        print(f"Loading documents from {RAW_DIR}...")
        chunks = ingest_all(RAW_DIR)
        print(f"Chunked into {len(chunks)} chunks")

        total = index_chunks(chunks, reset=reset)
        print(f"Indexed chunks: {total}" + (" (fresh build)" if reset else " (upserted)"))

        stats = index_stats()
        print("Index stats:")
        print(f"  total_chunks:         {stats['total_chunks']}")
        print(f"  source_type_counts:   {stats['source_type_counts']}")
        print(f"  business_unit_counts: {stats['business_unit_counts']}")
        return 0
    except Exception as exc:
        logging.getLogger("build_index").error("Index build failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
