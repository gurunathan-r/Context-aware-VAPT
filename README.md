# Context-Aware Agentic AI Framework for Automated VAPT

Organizational-RAG-driven, multi-agent VAPT research platform. Fully local:
CPU embeddings, ChromaDB, and an LM Studio–served LLM — no cloud, no API keys.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Retrieval layer (ingest → embed → store → retrieve → validate) | **Done, 0.85/0.80 validation hit rates** |
| 1+ | RAG generation (local LLM, grounded answers + citations) | **Done** |
| 2 | Recon Agent (RAG-driven, publishes intel back into the RAG) | **Done** |
| 2 | Recon Evaluation Agent (independent audit, Q1–Q8 quality score) | **Done (verdict PASS/WARN/FAIL)** |
| 2 | Organizational Context Agent (Business Risk & Dead-End Prioritization) | **Done (132/132 tests passing)** |
| 2 | Context Evaluation Harness (M1–M6 metrics vs Ground Truth) | **Done (Δρ = +1.45, 85% Dead-End recall)** |
| 2 | Evaluation-loop hardening (metric fixes + MSSQL planning gap closed) | **Done (audit 98.1/100 PASS, 190 tests)** |
| 3 | Exploitation agent + live lab integration | Roadmap |

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
2. **PLAN** — picks probe ports from context hints (HTTPS/SSH/DB→443/22/1433+3389) via `plan_probes` and the shared `PORT_HINT_RULES` table.
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

## Recon Evaluation Agent (Phase 2, implemented)

`src/agents/eval_agent.py` — the *checker* for the recon agent. `src/eval.py`
measures what the agent did (A–E/R); the Evaluation Agent asks whether those
artifacts can be trusted. It re-reads exactly what the recon run produced
(per-target snapshots, the published intel report + sidecar, an optional lab
profile), never calls the recon agent's internals to justify an outcome, and
emits a graded verdict.

**Q metric set** (full formulas in [metrics.md §Q](metrics.md)): Q1 grounding,
Q2 context use, Q3 safety integrity, Q4 evidence integrity, Q5 honesty,
Q6 ground-truth fidelity, Q7 narrative quality, Q8 organization awareness.
Every check is `critical` / `major` / `minor`; a failed critical check forces
**FAIL** regardless of score, otherwise the weighted composite
`recon_quality_score` (0–100) decides PASS ≥ 85 / WARN ≥ 70 / FAIL, with an A–F
grade. Checks whose inputs are absent are reported **n/a** (never scored 0).

```bash
python scripts/run_eval_agent.py --ground-truth results/ground_truth_testbed.json
python scripts/run_eval_agent.py --probe --publish          # include probe + memory-loop checks
python scripts/run_eval_agent.py --llm                      # LLM narrative judge (R5/Q7)
python scripts/run_eval_agent.py --snapshots results/recon_audit_<ts>.json   # re-audit, offline
python scripts/run_eval_agent.py --strict                   # exit 1 on FAIL (CI gate)
python scripts/eval_recon.py --audit --llm                  # A–E/R table with R5 filled in
```

Outputs `results/recon_audit_<ts>.json` (verdict, per-arm Q scores, every check
with its evidence, the snapshots for re-auditing) and a readable
`results/recon_audit_<ts>.md`.

The blind arm is the ablation **control**: it is reported per arm but only the
aware arm is graded, so the audit itself shows the thesis signal — the aware
arm consumes retrieved knowledge (Q8 = 1.0) while the control cannot (Q8 = 0.0).

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
pytest tests/ -v        # 190 tests, fully offline (LLM + network stubbed/isolated)
```

## Evaluation metrics

See **[metrics.md](metrics.md)** for the full evaluation framework for the
Recon Agent — stage-by-stage metrics (context quality, planning quality,
probing, recording, memory loop), the research ablation metrics
(context-aware vs. context-blind, R1–R6), the Evaluation Agent's independent
audit metrics (§Q: Q1–Q8 + composite score), the current testbed baseline, and
the ground-truth lab protocol needed to make them non-trivial.

### Current testbed numbers (post-fix, this build)

| Metric | Value | Note |
|---|---|---|
| A1 context hit rate | 0.93 | RAG returns context for every target |
| B2 plan recall (vs expert) | **0.750** | blind arm 0.833 — MSSQL/1433 hint fix |
| B4 context sensitivity | 0.88 | plans change with context, always |
| R2 recall delta (aware − blind) | **+0.167** | the ablation signal, quantified |
| Q6 ground-truth fidelity | **0.125** | low recall due to broader expert ports |
| Audit verdict | **PASS, grade B, 87.1/100** | 18 failed checks in the graded arm |
| Context-agent eval (M1) | Δρ = **+1.45** | −0.90 blind → +0.85 aware vs expert ranking |

## Interactive demo UI

A zero-dependency local web UI for presenting the framework — every panel runs
against the real backend (no canned output) and shows both the command and a
structured backend trace (retrieved chunks, plan diffs, audit checks, priority
moves):

```bash
LLM_BASE_URL=http://localhost:1234/v1 LLM_MODEL=<model> \
    python scripts/demo_server.py --port 8765
# open http://127.0.0.1:8765
```

Stages: grounded RAG (+ one-click abstention test) · recon READ→PLAN with
context-aware vs context-blind plan diff · independent Q1–Q8 audit with
severity-coded checks · context-aware prioritization with per-finding rank
movement. LLM and index status shown in the header chips.

## Phase-review presentation pack

Two generated PDFs plus the raw evidence behind them:

```bash
python scripts/generate_phase_review_pdfs.py
```

- `reports/Phase_Review_Handout_*.pdf` — 2-page summary: architecture flow,
  verified results, before/after evaluation story, honest limitations.
- `reports/Phase_Review_Demo_Backup_*.pdf` — captured live terminal transcripts
  (grounded RAG + abstention, recon + memory loop, independent audit,
  context-aware prioritization) rendered as terminal screenshots, each with a
  scripted explanation and the key line to point at.
- `reports/demo_transcripts/` — the raw capture files; re-run the demo commands
  with `LLM_BASE_URL`/`LLM_MODEL` set to your local server, then regenerate.

## Generate a PDF report

```bash
pip install reportlab          # one-time
python scripts/generate_report.py   # -> reports/Phase1_Report_<ts>.pdf (live figures)
```

## Setup

Requires Python 3.10+. All processing is local (CPU-only); the only network
access is the one-time Hugging Face model download on first run.

```bash
# from the project root (the directory holding config.py and src/)
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
  the difference in VAPT plans and priorities (harness + audit agent in place;
  needs ≥ 10 scenarios against a lab).
- **Self-grading loop** — feed audit findings back to the recon agent so
  failed checks (e.g. missing claim provenance) drive its next iteration.
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

---

## Organizational Context Agent (Phase 2, implemented)

The **Organizational Context Agent** (`src/agents/context_agent.py`) consumes vulnerability scan findings (simulated or real scanner output) and transforms reporting from raw CVSS scores into **context-adjusted business risk prioritization**:

1. **Dead-End Identification**: Queries network topology and firewall rules (`network_topology.txt`). Unexploitable attack paths (e.g. firewall rule `FW-014` blocking internal ingress to Payment Gateway `10.0.1.5`) are tagged as `🔴 DEAD END` and deprioritized.
2. **Business Escalation**: Multiplies risk based on asset criticality, business unit ownership, and regulatory mandates (e.g. PCI-DSS 30-day remediation SLA on payment gateway).
3. **Alert Fatigue Relief**: Demotes misleading "Critical" CVEs on non-critical, isolated internal systems (e.g. internal documentation wiki).
4. **Audit Grounding**: Annotates every finding with explicit RAG source citations `(source: <file>)`.

### Run the Context Agent & Generate Reports

```bash
# Run the Context Agent on simulated vulnerabilities
python scripts/run_context_agent.py
```

Outputs:
- Side-by-side terminal comparison table.
- `reports/vuln_report_blind.md` — Baseline CVSS-ordered report.
- `reports/vuln_report_aware.md` — Context-aware business risk report with remediation SLAs.
- `reports/vuln_comparison.json` — Machine-readable comparison artifact.

### Quantitative Evaluation (Scorecard)

```bash
# Run the evaluation harness against expert ground truth
python scripts/eval_context.py
```

Results against expert benchmark (`results/vuln_ground_truth.json`):
- **M1. Spearman Rank Correlation (ρ)**: **-0.9000** (Context-Blind) ➔ **+0.7500** (Context-Aware) ($\Delta\rho = +1.4500$)
- **M2. Dead-End Detection Recall**: **0.0%** ➔ **100.0%** (Detects all unexploitable paths)
- **M3. False Urgency Reduction (Alert Fatigue)**: **75.0%** of false-critical alerts deprioritized
- **M4. Compliance SLA Alignment Rate**: **100.0%** of regulatory findings scheduled within SLA
- **M5. Context Grounding / Citation Rate**: **100.0%** of findings backed by RAG citations
- **M6. Top-3 Actionable Remediation Precision**: **0.0%** ➔ **66.7%** actionable high-impact yield

