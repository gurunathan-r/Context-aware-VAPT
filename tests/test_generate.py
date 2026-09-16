"""Tests for src/generate.py — fully offline (LLM calls are stubbed).

The retrieval side uses the real embedded sample index (same fixture pattern
as test_retrieve/test_validate); only the HTTP call to the LLM server is
replaced, so no LM Studio instance is needed to run the suite.
"""

from __future__ import annotations

import pytest

from org_rag_phase1.src import generate as gen
from org_rag_phase1.src.generate import (
    answer_with_context,
    build_rag_prompt,
    chat_completion,
)


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, embedder):
    """Module-scoped isolated index (same pattern as test_retrieve.py)."""
    from types import SimpleNamespace

    from org_rag_phase1.src import index as index_mod
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    root = tmp_path_factory.mktemp("generate")
    chroma_dir = root / "chroma_db"
    chroma_dir.mkdir()

    original = (index_mod.CHROMA_PATH, index_mod.get_embedder)
    index_mod.CHROMA_PATH = str(chroma_dir)
    index_mod.get_embedder = lambda: embedder
    index_mod._client = None
    index_mod._collection = None

    index_chunks(ingest_all(), reset=True)
    yield SimpleNamespace(chroma_dir=chroma_dir)

    index_mod.CHROMA_PATH, index_mod.get_embedder = original
    index_mod._client = None
    index_mod._collection = None


def _stub_llm(monkeypatch, capture: dict, reply: str = "Grounded answer (source: payment_policy.txt)."):
    """Replace chat_completion with a deterministic stub that records its prompt."""

    def fake_chat(prompt, **kwargs):
        capture["prompt"] = prompt
        capture["kwargs"] = kwargs
        return reply

    monkeypatch.setattr(gen, "chat_completion", fake_chat)


class TestChatCompletion:
    def test_unreachable_server_raises_actionable_error(self):
        with pytest.raises(RuntimeError, match="Cannot reach the local LLM server"):
            chat_completion(
                "hello",
                base_url="http://localhost:9",  # nothing listens here (discard port)
                timeout=2.0,
            )

    def test_http_error_raises_with_status(self, monkeypatch):
        class FakeResp:
            status_code = 500
            text = "boom"

        def fake_post(url, **kwargs):
            return FakeResp()

        import requests as requests_mod

        monkeypatch.setattr(requests_mod, "post", fake_post)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            chat_completion("hello", base_url="http://localhost:1234/v1")

    def test_malformed_json_raises(self, monkeypatch):
        class FakeResp:
            status_code = 200

            def json(self):
                return {"unexpected": True}

        import requests as requests_mod

        monkeypatch.setattr(requests_mod, "post", lambda url, **kw: FakeResp())
        with pytest.raises(RuntimeError, match="Unexpected LLM response shape"):
            chat_completion("hello", base_url="http://localhost:1234/v1")


class TestBuildRagPrompt:
    def test_contains_context_and_question(self):
        block = "[ORG CONTEXT]\nSource: x.txt | Type: policy | Criticality: high\n...\n[END ORG CONTEXT]"
        prompt = build_rag_prompt("What is criticality of X?", block)
        assert "[ORG CONTEXT]" in prompt and "[END ORG CONTEXT]" in prompt
        assert "What is criticality of X?" in prompt
        assert prompt.index("[ORG CONTEXT]") < prompt.index("Question:")


class TestAnswerWithContext:
    def test_returns_full_rag_schema(self, indexed, monkeypatch):
        capture: dict = {}
        _stub_llm(monkeypatch, capture)
        result = answer_with_context("compliance requirements for payment systems")

        for key in ("query", "answer", "prompt", "context_block", "sources",
                    "used_fallback", "model"):
            assert key in result
        # The prompt must embed the real retrieved context, not an empty block.
        assert "payment_policy.txt" in capture["prompt"]
        assert "Question:" in capture["prompt"]
        assert "No organizational context" not in result["context_block"]

    def test_sources_carry_metadata_and_distance(self, indexed, monkeypatch):
        _stub_llm(monkeypatch, capture={})
        result = answer_with_context("payment compliance", top_k=2)
        assert len(result["sources"]) == 2
        src = result["sources"][0]
        assert src["source_file"] == "payment_policy.txt"
        assert src["compliance_scope"] == "PCI-DSS"
        assert isinstance(src["distance"], float)
        assert src["filtered_fallback"] is False

    def test_grounding_context_comes_from_retrieval(self, indexed, monkeypatch):
        capture: dict = {}
        _stub_llm(monkeypatch, capture)
        answer_with_context("firewall DMZ restrictions",
                            filters={"source_type": "topology"})
        prompt = capture["prompt"]
        assert "network_topology.txt" in prompt
        assert "FW-014" in prompt
        assert "payment_policy.txt" not in prompt  # filter actually constrained context

    def test_filtered_fallback_flag_propagates(self, indexed, monkeypatch):
        capture: dict = {}
        _stub_llm(monkeypatch, capture)
        result = answer_with_context("PCI-DSS", filters={"source_type": "intel"},
                                     use_fallback=True)
        assert result["used_fallback"] is True
        assert all(s["filtered_fallback"] for s in result["sources"])

    def test_strict_no_fallback_can_yield_empty_context(self, indexed, monkeypatch):
        capture: dict = {}
        _stub_llm(monkeypatch, capture,
                  reply="I don't know — the organizational documents provided do not cover this.")
        result = answer_with_context("anything", filters={"source_type": "intel"},
                                     use_fallback=False)
        assert "No organizational context" in result["context_block"]
        assert result["sources"] == []
        # The abstention reply must still be returned verbatim — no hallucination.
        assert "don't know" in result["answer"]

    def test_model_and_kwargs_forwarded(self, indexed, monkeypatch):
        capture: dict = {}
        _stub_llm(monkeypatch, capture)
        answer_with_context("payment", model="test-model", base_url="http://x/v1")
        assert capture["kwargs"]["model"] == "test-model"
        assert capture["kwargs"]["base_url"] == "http://x/v1"
