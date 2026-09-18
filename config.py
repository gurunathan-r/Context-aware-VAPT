"""Central configuration for the organizational RAG retrieval layer (Phase 1).

All tunable constants live here so that no src/ or scripts/ module hardcodes
paths or model parameters. Paths are expressed relative to the project root
(the directory containing this config.py) and are resolved against it via
pathlib, so commands work from any CWD:

    python scripts/build_index.py
    python -m pytest tests/ -v
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# Root = directory containing config.py (org_rag_phase1/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Input documents (policies, asset inventory, topology docs).
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Persistent ChromaDB storage — survives restarts (gitignored; rebuild anytime
# with `python scripts/build_index.py --reset`).
CHROMA_PATH: str = "data/chroma_db"

# Name of the single ChromaDB collection holding all org knowledge chunks.
COLLECTION_NAME: str = "org_knowledge"

# Hand-labeled validation test set used by src/validate.py.
TEST_QUERIES_PATH: Path = PROJECT_ROOT / "data" / "test_queries.json"

# Agent findings store: documents the Recon Agent writes back into the RAG as
# source_type="intel" live here (human-readable markdown, one file per run).
INTEL_DIR: Path = PROJECT_ROOT / "data" / "intel"

# Bug-bounty program scope store: fetched RFC 9116 security.txt policies land
# here as source_type="scope" documents (one per program/domain) and are indexed
# so agents can check legal scope *before* planning or probing.
SCOPE_DIR: Path = PROJECT_ROOT / "data" / "scope"

# Fetching a program's scope is the one deliberate network call this project
# makes (besides the one-time embedding model download). Bounded and local-only.
SECURITY_TXT_TIMEOUT: float = 10.0  # seconds per HTTP attempt
SCOPE_USER_AGENT: str = "org-rag-vapt-recon/0.1 (local research; RFC 9116 security.txt)"

# Default recon targets when none are passed on the CLI.
RECON_DEFAULT_TARGETS: list[str] = [
    "10.0.1.5", "10.0.2.10", "10.0.3.20", "10.0.4.30", "10.0.5.40",
    "10.0.6.50", "10.0.7.60", "10.0.8.70", "10.0.9.80", "10.0.10.90",
    "10.0.11.100", "10.0.12.110", "10.0.13.120", "10.0.14.130", "10.0.15.140"
]

# ---------------------------------------------------------------------------
# Embedding model (CPU-only, fully local — no external API calls)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence-transformers checkpoint (HF Hub), 384-dim

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # characters of overlap between consecutive chunks

# Splitter fallbacks: paragraph -> sentence -> whitespace -> character.
# "。" and "；" are full-width CJK sentence punctuation, included per spec so
# CJK documents also split on sentence boundaries.
SEPARATORS = ["\n\n", "\n", "。", "；", " ", ""]

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
BATCH_SIZE = 64  # embedding batch size
COSINE_SPACE = "cosine"  # ChromaDB space; distances are cosine distance in [0, 2]

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
TOP_K_DEFAULT = 3

# ---------------------------------------------------------------------------
# Metadata enums (closed vocabularies enforced at ingest time)
# ---------------------------------------------------------------------------
SOURCE_TYPES = ["asset", "policy", "intel", "topology", "scope"]
BUSINESS_UNITS = ["finance", "hr", "engineering", "unknown"]
CRITICALITY_LEVELS = ["high", "medium", "low"]
AUTHORITY_LEVELS = ["mandatory", "advisory", "informational"]
COMPLIANCE_SCOPES = ["PCI-DSS", "GDPR", "none"]

# ChromaDB only persists metadata values of type str, int, float or bool; these
# substitutes are used by src/index.py when sanitizing values (None -> "unknown").
METADATA_NULL_SUBSTITUTE: str = "unknown"

# ---------------------------------------------------------------------------
# Generation (the "G" in RAG) — Pluggable LLM behind an OpenAI-compatible API.
# ---------------------------------------------------------------------------
# Works with any OpenAI-compatible backend:
#   - Ollama (default): http://localhost:11434/v1, model qwen3.5:9b-q4_K_M
#   - LM Studio:        http://localhost:1234/v1
#   - vLLM / llama.cpp / Cloud APIs (Gemini, OpenAI, etc.)
# All parameters can be overridden via environment variables or CLI arguments.
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5:9b-q4_K_M")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "180.0"))
RAG_TOP_K: int = TOP_K_DEFAULT         # chunks retrieved to build the context block
