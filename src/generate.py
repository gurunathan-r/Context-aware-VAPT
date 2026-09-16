"""RAG generation: retrieve organizational context, then answer with a local LLM.

This module completes the "G" in RAG on top of the Phase 1 retrieval layer:

    query ──► retrieve_org_context() ──► format_context_for_prompt()
              (ChromaDB cosine search)        │
                                              ▼
                              local LLM (OpenAI-compatible API, e.g. LM Studio)
                                              │
                                              ▼
                              grounded answer + cited sources

Everything stays local: the LLM is served from localhost (LM Studio by
default) with no cloud calls and no API keys. Any OpenAI-compatible server
works — point LLM_BASE_URL at it.

Design decision — strict grounding: the system prompt forbids answering from
the model's own knowledge and instructs it to say so when the context is
insufficient. This keeps the research reproducible: answers must be traceable
to indexed organizational documents, not to the base model's parameters.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from org_rag_phase1.config import (
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    RAG_TOP_K,
)
from org_rag_phase1.src.retrieve import (
    format_context_for_prompt,
    retrieve_org_context,
    retrieve_with_fallback,
)

logger = logging.getLogger(__name__)


def _strip_reasoning_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks some models emit.

    Safety net for Qwen3-style models: with some serving stacks the reasoning
    leaks into the content field instead of a separate reasoning field.
    """
    while "<think>" in text and "</think>" in text:
        start = text.index("<think>")
        end = text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


# When True (default), a filtered search that comes back empty falls back to an
# unfiltered search (results tagged filtered_fallback=True) instead of leaving
# the LLM with an empty [ORG CONTEXT] block.
USE_FALLBACK_RETRIEVAL = True

SYSTEM_PROMPT = """\
You are a security-analysis assistant for an organization. You answer strictly \
from the [ORG CONTEXT] block provided with each question.

Rules:
1. Use ONLY facts present in [ORG CONTEXT]. Do not use outside knowledge.
2. Every factual claim must cite its source file in the form (source: <file>).
3. If the context does not contain the answer, reply exactly: \
"I don't know — the organizational documents provided do not cover this."
4. Be concise and concrete. Quote criticality levels, compliance scopes, \
IP addresses, and rule ids exactly as written in the context.
5. Output the final answer only (a few sentences or short bullets). \
Do not narrate your reasoning, planning, or the question back."""


def build_rag_prompt(query: str, context_block: str) -> str:
    """Compose the final user prompt from query + retrieved context.

    Args:
        query: The user's natural-language question.
        context_block: Output of format_context_for_prompt().

    Returns:
        The full user-message string sent to the LLM.
    """
    return (
        f"{context_block}\n\n"
        f"Question: {query}\n"
        "Answer using only the context above, citing (source: <file>) for "
        "each factual claim."
    )


def chat_completion(
    prompt: str,
    *,
    model: str = LLM_MODEL,
    base_url: str = LLM_BASE_URL,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: float = LLM_TIMEOUT_SECONDS,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the text.

    Works with LM Studio (`lms server start` on localhost:1234), llama.cpp
    server, vLLM, Ollama's /v1 endpoint, etc.

    Args:
        prompt: The user message (typically build_rag_prompt output).
        model: Model id as reported by the server (e.g. ``qwythos-9b``).
        base_url: API root, e.g. ``http://localhost:1234/v1``.
        system_prompt: System message steering grounded behavior.
        temperature: Sampling temperature (low for factual QA).
        max_tokens: Generation cap.
        timeout: Request timeout in seconds (local models can be slow).

    Returns:
        The assistant's reply text.

    Raises:
        RuntimeError: On connection failure or non-200 response, with context
            for debugging (status body included). Never silently ignored.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach the local LLM server at {base_url}. "
            "Start LM Studio and run `lms server start` (or point "
            "config.LLM_BASE_URL at another OpenAI-compatible server)."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"LLM request to {url} timed out after {timeout}s "
            "(local models can be slow on CPU; try a smaller model)."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM server returned HTTP {response.status_code} for {url}: "
            f"{response.text[:500]}"
        )

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"].strip()
        return _strip_reasoning_blocks(content)
    except (ValueError, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected LLM response shape from {url}: {str(body)[:500]}"
        ) from exc


def answer_with_context(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = RAG_TOP_K,
    *,
    model: str = LLM_MODEL,
    base_url: str = LLM_BASE_URL,
    use_fallback: bool = USE_FALLBACK_RETRIEVAL,
) -> dict[str, Any]:
    """Full RAG loop for one query: retrieve → augment → generate.

    Downstream agents call this when they need a natural-language answer
    grounded in organizational documents, or call retrieve_org_context()
    directly when they only need the raw evidence.

    Args:
        query: Natural-language question, e.g.
            "What are the compliance requirements for payment systems?".
        filters: Optional ChromaDB metadata filters, e.g.
            ``{"source_type": "asset"}`` (see retrieve_org_context).
        top_k: Number of chunks to retrieve for the context block.
        model: LLM model id on the local server.
        base_url: OpenAI-compatible API root.
        use_fallback: If True (default), use retrieve_with_fallback so an
            over-narrow filter still yields context (tagged
            ``filtered_fallback``); if False, strict filtered retrieval only.

    Returns:
        {
          "query": str,
          "answer": str,               # LLM reply (grounded, with citations)
          "prompt": str,               # exact prompt sent — full auditability
          "context_block": str,        # [ORG CONTEXT] block built from hits
          "sources": [ {...}, ... ],   # per-chunk retrieval metadata + distance
          "used_fallback": bool,       # True if unfiltered retry occurred
          "model": str,
        }

    Raises:
        RuntimeError: If the LLM server is unreachable or errors.
        ValueError: If the query is empty or filters are malformed.
    """
    if use_fallback:
        results = retrieve_with_fallback(query, filters=filters, top_k=top_k)
    else:
        results = retrieve_org_context(query, filters=filters, top_k=top_k)

    used_fallback = bool(results) and bool(results[0].get("filtered_fallback", False))
    if not results:
        logger.warning("No context retrieved for query %r — LLM must abstain.", query)

    context_block = format_context_for_prompt(results)
    prompt = build_rag_prompt(query, context_block)
    answer = chat_completion(prompt, model=model, base_url=base_url)

    sources = [
        {
            "source_file": r.get("source_file"),
            "source_type": r.get("source_type"),
            "asset_criticality": r.get("asset_criticality"),
            "compliance_scope": r.get("compliance_scope"),
            "authority_level": r.get("authority_level"),
            "distance": r.get("distance"),
            "filtered_fallback": r.get("filtered_fallback", False),
        }
        for r in results
    ]
    logger.info(
        "RAG answer generated for %r (chunks=%d, fallback=%s, model=%s)",
        query, len(results), used_fallback, model,
    )
    return {
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "context_block": context_block,
        "sources": sources,
        "used_fallback": used_fallback,
        "model": model,
    }
