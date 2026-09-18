"""Shared pytest fixtures for Phase 1 tests.

Strategy:
    - All fixtures build a tiny, isolated vector store under tmp_path so tests
      never touch the real data/chroma_db and run fully offline.
    - The embedding model is loaded once per session (loading is the slow part;
      encoding itself is fast for tiny inputs) and monkeypatched into src.index,
      so every collection operation uses the same cached model without
      re-downloading or re-loading per test.
    - Deterministic offline behavior: the model is instantiated with
      local_files_only=True; tests fail fast if the checkpoint is not cached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the org_rag_phase1 package importable. tests/ lives two levels below the
# project root, and the project root holds the tracked path-shim package
# org_rag_phase1/ that maps org_rag_phase1.* onto this copy of the source tree.
# PROJECT_ROOT must come FIRST on sys.path: its parent may hold a sibling
# checkout that is also named org_rag_phase1 and would otherwise win.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
assert sys.path[0] == str(PROJECT_ROOT)  # shim beats any sibling org_rag_phase1/


@pytest.fixture(scope="session")
def embedder():
    """Session-wide sentence-transformers model (CPU, local files only)."""
    from sentence_transformers import SentenceTransformer

    from org_rag_phase1.config import EMBEDDING_MODEL

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)


@pytest.fixture()
def index_env(tmp_path, monkeypatch, embedder):
    """Isolated index environment: temp Chroma dir + cached embedder.

    Yields a SimpleNamespace with the temp chroma path. Resets src.index module
    state before and after so the persistent project index is never touched.
    """
    from types import SimpleNamespace

    from org_rag_phase1.src import index as index_mod

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(index_mod, "CHROMA_PATH", str(chroma_dir))
    monkeypatch.setattr(index_mod, "get_embedder", lambda: embedder)

    index_mod._client = None
    index_mod._collection = None

    env = SimpleNamespace(chroma_dir=chroma_dir)
    yield env

    index_mod._client = None
    index_mod._collection = None


@pytest.fixture()
def indexed_env(index_env):
    """An index_env pre-populated with the sample documents from data/raw/."""
    from org_rag_phase1.src.index import index_chunks
    from org_rag_phase1.src.ingest import ingest_all

    from org_rag_phase1.config import RAW_DIR

    index_chunks(ingest_all(RAW_DIR), reset=True)
    return index_env
