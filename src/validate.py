"""Validation harness for retrieval quality.

Measures whether the index actually answers the questions agents will ask:
each hand-labeled query lists the source files and keywords a *correct*
retrieval should surface. The harness records hit/miss per query and aggregates
hit rates against PASS thresholds (>= 0.80 for both metrics by default).

Run ad hoc:
    python -c "from org_rag_phase1.src.validate import run_validation, \
load_test_queries, print_report; from org_rag_phase1.config import \
TEST_QUERIES_PATH; print_report(run_validation(load_test_queries(str(TEST_QUERIES_PATH))))"
"""

import json
import logging
from pathlib import Path
from typing import Any

from org_rag_phase1.config import TEST_QUERIES_PATH, TOP_K_DEFAULT
from org_rag_phase1.src.retrieve import retrieve_org_context

logger = logging.getLogger(__name__)

# Aggregate hit rates at or above these thresholds count as PASS.
PASS_THRESHOLD = 0.80


def load_test_queries(path: str | Path = TEST_QUERIES_PATH) -> list[dict[str, Any]]:
    """Load the hand-labeled validation test set.

    Args:
        path: Path to a JSON file whose top-level value is a list of objects:
            ``{"query": str, "filters": dict | None,
               "expected_source_files": [str], "expected_keywords": [str]}``.

    Returns:
        The parsed list of test-query dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is not a non-empty list, or an entry is missing
            required keys.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Test query file not found: {path.resolve()}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ValueError(f"{path.name} must contain a non-empty JSON list.")

    required = {"query", "filters", "expected_source_files", "expected_keywords"}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(
                f"{path.name}[{i}] must be an object with keys {sorted(required)}."
            )
    return data


def run_validation(
    test_queries: list[dict[str, Any]], top_k: int = TOP_K_DEFAULT
) -> dict[str, Any]:
    """Run every test query against the live index and score retrieval quality.

    For each query:
        - calls retrieve_org_context (with the query's filters),
        - source_file hit: any expected_source_files entry appears among the
          top_k results' source_file metadata,
        - keyword hit: any expected_keywords entry appears (case-insensitive
          substring) in any retrieved chunk_text.

    Args:
        test_queries: Entries in the load_test_queries format.
        top_k: How many chunks to retrieve per query (default 3).

    Returns:
        Aggregate report:
        {
          "total_queries": int,
          "source_file_hit_rate": float in [0, 1],
          "keyword_hit_rate": float in [0, 1],
          "per_query_results": [
              {"query": ..., "filters": ..., "hit_source_file": bool,
               "hit_keyword": bool, "matched_source_files": [...],
               "matched_keywords": [...], "retrieved_source_files": [...]}
          ]
        }

    Raises:
        ValueError: If ``test_queries`` is empty.
    """
    if not test_queries:
        raise ValueError("run_validation called with an empty test query list.")

    per_query: list[dict[str, Any]] = []
    source_hits = 0
    keyword_hits = 0

    for entry in test_queries:
        query = entry["query"]
        filters = entry.get("filters")
        expected_files = list(entry.get("expected_source_files") or [])
        expected_keywords = list(entry.get("expected_keywords") or [])

        results = retrieve_org_context(query, filters=filters, top_k=top_k)
        retrieved_files = [r.get("source_file") for r in results]
        haystack = "\n".join(str(r.get("chunk_text", "")) for r in results).lower()

        matched_files = [f for f in expected_files if f in retrieved_files]
        matched_keywords = [k for k in expected_keywords if str(k).lower() in haystack]

        hit_source = bool(matched_files)
        # No expected keywords => nothing to check => vacuously a hit (the
        # keyword metric should not penalize queries that only assert provenance).
        hit_keyword = not expected_keywords or bool(matched_keywords)
        source_hits += hit_source
        keyword_hits += hit_keyword

        per_query.append(
            {
                "query": query,
                "filters": filters,
                "hit_source_file": hit_source,
                "hit_keyword": hit_keyword,
                "matched_source_files": matched_files,
                "matched_keywords": matched_keywords,
                "retrieved_source_files": retrieved_files,
            }
        )
        logger.debug(
            "query=%r source_hit=%s keyword_hit=%s", query, hit_source, hit_keyword
        )

    total = len(test_queries)
    return {
        "total_queries": total,
        "source_file_hit_rate": source_hits / total,
        "keyword_hit_rate": keyword_hits / total,
        "per_query_results": per_query,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable pass/fail validation summary.

    Thresholds: source_file_hit_rate >= 0.80 and keyword_hit_rate >= 0.80.

    Args:
        report: Report as returned by run_validation.
    """
    total = report.get("total_queries", 0)
    src_rate = report.get("source_file_hit_rate", 0.0)
    kw_rate = report.get("keyword_hit_rate", 0.0)

    print("=" * 70)
    print("RETRIEVAL VALIDATION REPORT")
    print("=" * 70)
    print(f"Total queries:          {total}")
    print(f"Source-file hit rate:   {src_rate:.2f}  "
          f"({'PASS' if src_rate >= PASS_THRESHOLD else 'FAIL'} — threshold {PASS_THRESHOLD:.2f})")
    print(f"Keyword hit rate:       {kw_rate:.2f}  "
          f"({'PASS' if kw_rate >= PASS_THRESHOLD else 'FAIL'} — threshold {PASS_THRESHOLD:.2f})")
    print("-" * 70)
    for r in report.get("per_query_results", []):
        status = "OK " if (r["hit_source_file"] and r["hit_keyword"]) else "MISS"
        print(f"[{status}] {r['query']}")
        if r.get("filters"):
            print(f"       filters: {r['filters']}")
        print(f"       retrieved: {r['retrieved_source_files']}")
        if not r["hit_source_file"]:
            print(f"       expected source file(s): {r.get('matched_source_files') or 'n/a'}")
        if not r["hit_keyword"]:
            print("       NOTE: no expected keyword found in retrieved chunks")
    print("=" * 70)
