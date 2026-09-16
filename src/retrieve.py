"""Retrieval interface for downstream agents.

This module is the contract between the org-knowledge store and the rest of
the project: future recon/analysis/planning agents call retrieve_org_context()
(or retrieve_with_fallback()) and receive self-describing context records, or
format_context_for_prompt() to inject results straight into an LLM prompt.

Call pattern for agents:

    from org_rag_phase1.src.retrieve import (
        retrieve_org_context, format_context_for_prompt,
    )

    results = retrieve_org_context(
        "What must we protect on the payment network?",
        filters={"compliance_scope": "PCI-DSS"},
        top_k=5,
    )
    prompt_block = format_context_for_prompt(results)
"""

import logging
from typing import Any

from org_rag_phase1.config import TOP_K_DEFAULT
from org_rag_phase1.src.index import get_collection, get_embedder

logger = logging.getLogger(__name__)

# Keys copied from chunk metadata into each returned record. ``distance`` is
# appended separately from the query response.
_RESULT_META_KEYS = (
    "source_type",
    "business_unit",
    "asset_criticality",
    "compliance_scope",
    "authority_level",
    "source_file",
    "chunk_id",
    "chunk_index",
    "total_chunks",
)

# Operators understood by the filter builder (see _build_where).
_SUPPORTED_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}


def _build_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a simple filters dict into a ChromaDB ``where`` clause.

    Rules:
        - ``{"field": value}`` becomes ``{"field": {"$eq": value}}``.
        - ``{"field": [v1, v2]}`` becomes ``{"field": {"$in": [v1, v2]}}``.
        - Explicit operator dicts pass through after validation, e.g.
          ``{"asset_criticality": {"$in": ["high", "medium"]}}``.
        - Multiple fields combine with ``$and`` (ChromaDB allows only one
          top-level operator).

    Args:
        filters: Mapping of metadata field to value, list value, or operator
            dict. Empty dict means "no filtering".

    Returns:
        A ChromaDB where clause, or None when ``filters`` is empty.

    Raises:
        ValueError: If a filter uses an unsupported operator or a malformed
            operator dict.
    """
    if not filters:
        return None

    clauses: list[dict[str, Any]] = []
    for field, value in filters.items():
        if isinstance(value, dict):
            unknown = set(value) - _SUPPORTED_OPS
            if unknown or not value:
                raise ValueError(
                    f"Filter for '{field}' uses unsupported operator(s): "
                    f"{sorted(unknown) if unknown else '(empty)'}; supported: "
                    f"{sorted(_SUPPORTED_OPS)}"
                )
            clauses.append({field: value})
        elif isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                raise ValueError(f"Filter for '{field}' has an empty list value.")
            clauses.append({field: {"$in": list(value)}})
        else:
            clauses.append({field: {"$eq": value}})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_org_context(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = TOP_K_DEFAULT,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant organizational context chunks for a query.

    This is the primary entry point for downstream agents. It embeds the query
    with the same model used at indexing time (all-MiniLM-L6-v2) and performs a
    cosine similarity search over the persistent collection.

    Args:
        query: Natural-language question or task description, e.g.
            "compliance requirements for payment systems".
        filters: Optional metadata filters applied server-side by ChromaDB,
            e.g. ``{"source_type": "asset"}`` or
            ``{"compliance_scope": ["PCI-DSS", "GDPR"]}`` (list => $in).
        top_k: Maximum number of chunks to return (default 3).

    Returns:
        List of result dicts (most relevant first), each with keys:
        ``chunk_text``, ``source_type``, ``business_unit``,
        ``asset_criticality``, ``compliance_scope``, ``authority_level``,
        ``source_file``, ``chunk_id``, ``chunk_index``, ``total_chunks``,
        ``distance`` (cosine distance; smaller = more similar).
        Empty list when the collection is empty or nothing matches the filters.

    Raises:
        ValueError: If ``query`` is empty or ``filters`` is invalid.
    """
    if not query or not query.strip():
        raise ValueError("retrieve_org_context called with an empty query.")

    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Collection is empty — run scripts/build_index.py first.")
        return []

    where = _build_where(filters or {})
    embedder = get_embedder()
    query_embedding = embedder.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    )[0]

    response = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _format_response(response)


def _format_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a ChromaDB query response into agent-friendly result dicts."""
    results: list[dict[str, Any]] = []
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        record: dict[str, Any] = {"chunk_text": doc}
        for key in _RESULT_META_KEYS:
            record[key] = meta.get(key)
        record["distance"] = dist
        results.append(record)
    return results


def retrieve_with_fallback(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = TOP_K_DEFAULT,
    min_results: int = 1,
) -> list[dict[str, Any]]:
    """Retrieve with metadata filters, falling back to unfiltered search.

    Agents often start with a narrow hypothesis (e.g. only ``source_type``
    ``asset``). If the filtered search cannot produce at least ``min_results``
    hits, this retries without filters so the agent still receives context,
    marking fallback results with ``"filtered_fallback": True`` so callers can
    distinguish "confirmed by filtered evidence" from "best-effort global
    evidence".

    Args:
        query: Natural-language query (see retrieve_org_context).
        filters: Optional metadata filters for the first attempt.
        top_k: Maximum number of chunks to return.
        min_results: Minimum acceptable result count before falling back.

    Returns:
        Result dicts in the retrieve_org_context format; each carries
        ``filtered_fallback`` (True on fallback results, False otherwise).
        Results from a successful filtered query keep ``filtered_fallback``:
        False.
    """
    results = retrieve_org_context(query, filters=filters, top_k=top_k)
    if len(results) >= min_results:
        for r in results:
            r["filtered_fallback"] = False
        return results

    if filters:
        logger.info(
            "Filtered retrieval returned %d result(s) (< min_results=%d); "
            "falling back to unfiltered search.",
            len(results),
            min_results,
        )
    fallback = retrieve_org_context(query, filters=None, top_k=top_k)
    for r in fallback:
        r["filtered_fallback"] = True
    return fallback


def format_context_for_prompt(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a prompt-ready [ORG CONTEXT] block.

    Downstream agents call this right before invoking an LLM so each retrieved
    chunk carries its provenance (source file, type, criticality, authority,
    compliance scope) inline with the text.

    Args:
        results: Records as returned by retrieve_org_context /
            retrieve_with_fallback.

    Returns:
        A formatted string:

            [ORG CONTEXT]
            Source: <source_file> | Type: <source_type> | Criticality: <asset_criticality>
            Authority: <authority_level> | Compliance: <compliance_scope>
            <chunk_text>
            ---
            [END ORG CONTEXT]

        Empty results produce a block stating that no organizational context
        was found (so the agent never silently proceeds without context).
    """
    if not results:
        return (
            "[ORG CONTEXT]\nNo organizational context matched this query.\n"
            "[END ORG CONTEXT]"
        )

    lines: list[str] = ["[ORG CONTEXT]"]
    for r in results:
        lines.append(
            f"Source: {r.get('source_file', 'unknown')} | "
            f"Type: {r.get('source_type', 'unknown')} | "
            f"Criticality: {r.get('asset_criticality', 'unknown')}"
        )
        lines.append(
            f"Authority: {r.get('authority_level', 'unknown')} | "
            f"Compliance: {r.get('compliance_scope', 'unknown')}"
        )
        lines.append(str(r.get("chunk_text", "")))
        lines.append("---")
    lines.append("[END ORG CONTEXT]")
    return "\n".join(lines)
