<<<<<<< HEAD
# Context-Aware Agentic AI Framework for Automated VAPT

Organizational-RAG-driven, multi-agent VAPT research platform. Fully local:
CPU embeddings, ChromaDB, and an LM Studio–served LLM — no cloud, no API keys.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Retrieval layer (ingest → embed → store → retrieve → validate) | **Done, 1.00/1.00 validation hit rates** |
| 1+ | RAG generation (local LLM, grounded answers + citations) | **Done** |
| 2 | Recon Agent (RAG-driven, publishes intel back into the RAG) | **Done** |
| 2 | Vulnerability Analysis Agent | To implement |
| 2 | Prioritization / Planning Agent | To implement |
| 3 | Exploitation agent + ablation study | To implement |

## Phase 1 — Organizational RAG retrieval layer

A fully local, reproducible RAG foundation: ingests organizational documents
(policies, asset inventories, topology docs), chunks them semantically with
rich metadata, embeds them (`all-MiniLM-L6-v2`, CPU), and stores them in a
persistent ChromaDB collection with cosine space. One clean interface —
`retrieve_org_context()` — serves everything downstream.

| Implemented | Details |
|---|---|
| `src/ingest.py` | Sidecar `.meta.json` enforcement (hard error if missing), 1000/200 chunking, deterministic SHA-256 chunk ids |
| `src/index.py` | Lazy embedder cache, PersistentClient, batch-64 upserts, metadata sanitizer, `index_stats()` |
| `src/retrieve.py` | Filters (`$eq`/`$in`/`$and`), `retrieve_with_fallback`, `format_context_for_prompt` |
| `src/validate.py` | Labeled-query harness, source-file & keyword hit rates, PASS ≥ 0.80 |
| `scripts/build_index.py` | `--reset`, idempotent, exit codes |
| `scripts/demo_retrieval.py` | 3 example queries with prompt-ready output |

### Metadata schema

| Field | Values | Why it exists |
|---|---|---|
| `source_type` | `asset`, `policy`, `intel`, `topology`, `scope` | Scope queries by evidence kind (`intel` = agent-written findings, `scope` = fetched bug-bounty program scope) |
| `business_unit` | `finance`, `hr`, `engineering`, `unknown` | Accountable owner; escalates finance/HR findings |
| `asset_criticality` | `high`, `medium`, `low` | Core context signal — medium vuln on a payment gateway outranks high vuln on a wiki |
| `compliance_scope` | `PCI-DSS`, `GDPR`, `none` | Regulated scope changes remediation priority |
| `authority_level` | `mandatory`, `advisory`, `informational` | Mandatory rules constrain what agents may plan |
| `source_file` | file name | Provenance — every fact auditable to a document |
| `chunk_id` / `chunk_index` / `total_chunks` | SHA-256 prefix / ints | Deterministic identity, idempotent rebuilds |

## RAG generation — local LLM (LM Studio)

```bash
lms server start
python scripts/rag_query.py "What are the compliance requirements for payment systems?"
python scripts/rag_query.py "Criticality of 10.0.1.5?" --filter source_type=asset
```

- Any OpenAI-compatible local server works (`config.LLM_BASE_URL`); default model `config.LLM_MODEL`.
- `src/generate.py` — `answer_with_context()` returns `{answer, prompt, context_block, sources, used_fallback, model}`.
- **Grounding contract:** system prompt forbids model-memory answers, requires `(source: <file>)` citations, and mandates the exact abstention sentence when context is insufficient. Verified: unknown questions produce *"I don't know — the organizational documents provided do not cover this."* — never hallucinated answers.
- `<think>` reasoning blocks are stripped automatically (Qwen-style models).

## Recon Agent (Phase 2, implemented)

`src/agents/recon.py` — first agent on the loop. Per target:

1. **READ** — pulls asset/topology/policy context from the RAG (`gather_context`).
2. **PLAN** — picks probe ports from context hints (HTTPS/SSH/DB mentions) via `plan_probes`.
3. **PROBE** — safe TCP-connect checks only; **hard guardrail: RFC1918/documentation targets only** unless `--allow-public` (documented override).
4. **RECORD** — structured `Finding` records (category, detail, severity hint, timestamp).
5. **PUBLISH** — writes `data/intel/recon_findings_<ts>.md` + sidecar (`source_type="intel"`), indexes it → findings become retrievable organizational memory for the next run/agent.

```bash
python scripts/run_recon.py                    # default testbed targets
python scripts/run_recon.py --targets 10.0.1.5 --no-index   # dry-run (report only)
```

Verified loop: after a run, `rag_query.py "What did the recon agent find about 10.0.1.5?"`
retrieves the agent's own intel chunks and answers with citations from them.

### Bug-bounty scope awareness (minimal viable)

A publication-aware recon: the agent can learn **known legal scope** before it
plans or probes, from standards-based, keyless RFC 9116 `security.txt` files —
no HackerOne/Bugcrowd API keys, no scraping.

```bash
python scripts/fetch_scope.py github.com            # save data/scope/scope_github.com.md + index
python scripts/run_recon.py --fetch-scope github.com --allow-public --targets api.github.com
```

- **FETCH** — `src/scope.py::fetch_security_txt()`: https then http, `/.well-known/security.txt` then `/security.txt`; extracts Canonical/Contact/Expires.
- **SAVE** — `save_scope_doc()` writes markdown + sidecar (`source_type="scope"`, `authority_level="mandatory"` — scope is a legal boundary).
- **CHECK** — `check_scope()`: proven pure offline; matches hostnames, wildcards (`*.example.com`), and CIDR (`10.0.0.0/8`) against indexed policy chunks.

Gating contract (the important part): scope checks **never enable** probing —
the RFC1918 guardrail / `--allow-public` always applies first. Scope only ever
*restricts*: a host a published policy explicitly covers gets tagged
`in_scope`; a host the policy explicitly talks about but does not cover yields
a blocking record and **zero probes**; hosts with no matching policy keep
today's behavior. Covered by 21 offline tests (mocked network).

### Example: how a future Planning Agent will use this

```python
from org_rag_phase1.src.retrieve import retrieve_org_context, format_context_for_prompt

results = retrieve_org_context(
    "payment gateway 10.0.1.5 criticality and compliance scope",
    filters={"source_type": "asset"}, top_k=3,
)
asset_criticality = results[0]["asset_criticality"]   # "high"
compliance_scope  = results[0]["compliance_scope"]    # "PCI-DSS"

BASE_RISK = {"low": 1.0, "medium": 2.0, "high": 3.0}
risk = base_cvss_score * BASE_RISK[asset_criticality]
if compliance_scope != "none":
    risk *= 1.5   # regulated scope raises priority

prompt = f"""{format_context_for_prompt(results)}

You are the Planning Agent. Rank the following candidate attack paths
against 10.0.1.5 using the organizational context above..."""
```

## Run tests

```bash
pytest tests/ -v        # 108 tests, fully offline (LLM + network stubbed/isolated)
```

## Evaluation metrics

See **[metrics.md](metrics.md)** for the full evaluation framework for the
Recon Agent — stage-by-stage metrics (context quality, planning quality,
probing, recording, memory loop), the research ablation metrics
(context-aware vs. context-blind, R1–R6), the current testbed baseline, and
the ground-truth lab protocol needed to make them non-trivial.

## Generate a PDF report

```bash
pip install reportlab          # one-time
python scripts/generate_report.py   # -> reports/Phase1_Report_<ts>.pdf (live figures)
```

## Setup

Requires Python 3.10+. All processing is local (CPU-only); the only network
access is the one-time Hugging Face model download on first run.

```bash
cd org_rag_phase1
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
```

## To implement (roadmap)

- **Vulnerability Analysis Agent** — consume recon intel, map findings to
  candidate vulnerabilities (banner/service fingerprinting, CVE hypotheses).
- **Prioritization / Planning Agent** — LLM-driven attack-path ranking using
  `asset_criticality`, `compliance_scope`, `authority_level` (the Phase 1
  metadata is built exactly for this); context-weighted risk scoring.
- **Exploitation agent** — safe, lab-only exploitation of planned paths.
- **Ablation study** — the research deliverable: run the pipeline with and
  without organizational context (context-blind vs. context-aware) and measure
  the difference in VAPT plans and priorities.
- **Recon improvements** — service-version fingerprinting from banners;
  topology-aware probe planning (firewall-rule reasoning via RAG); differential
  recon (diff vs. last run); dedup/cleanup of repeated intel reports.
- **Retrieval upgrades** — reranking/hybrid (BM25) retrieval; MRR/nDCG metrics
  in the validation harness; incremental document updates (no full rebuild).

## Known limitations

- **Single flat collection** — no per-agent scoping beyond metadata filters.
- **No reranking or hybrid search** — pure dense cosine retrieval.
- **Hit-rate validation only** — no MRR/nDCG ranking quality metrics yet.
- **`.txt`/`.md` only** — no PDF/DOCX/HTML parsing.
- **Generic embeddings** — not tuned for security jargon (CVE ids, exploit names).
- **Static org snapshot** — changed documents require a rebuild (`--reset`).
- **Recon probes are TCP-connect only** — no service fingerprinting, no
  vulnerability detection yet (by design for Phase 2 start).
- **Single-tenant** — no multi-tenancy or retrieval access-control layer.
=======
# Context-aware-VAPT
>>>>>>> origin/main
