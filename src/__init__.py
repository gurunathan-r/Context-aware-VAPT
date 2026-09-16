"""Organizational RAG retrieval layer — Phase 1.

Modules:
    ingest:   load raw documents + sidecar metadata, chunk semantically.
    index:    embed chunks (all-MiniLM-L6-v2) and persist to ChromaDB.
    retrieve: the retrieval interface downstream agents will call.
    validate: retrieval quality harness against a hand-labeled test set.
"""

from org_rag_phase1.src.ingest import chunk_document, ingest_all, load_documents
from org_rag_phase1.src.index import (
    get_collection,
    get_embedder,
    index_chunks,
    index_stats,
)
from org_rag_phase1.src.retrieve import (
    format_context_for_prompt,
    retrieve_org_context,
    retrieve_with_fallback,
)
from org_rag_phase1.src.validate import (
    load_test_queries,
    print_report,
    run_validation,
)
from org_rag_phase1.src.generate import (
    answer_with_context,
    build_rag_prompt,
    chat_completion,
)

__all__ = [
    "chunk_document",
    "ingest_all",
    "load_documents",
    "get_collection",
    "get_embedder",
    "index_chunks",
    "index_stats",
    "format_context_for_prompt",
    "retrieve_org_context",
    "retrieve_with_fallback",
    "load_test_queries",
    "print_report",
    "run_validation",
    "answer_with_context",
    "build_rag_prompt",
    "chat_completion",
]
