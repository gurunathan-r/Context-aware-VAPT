# Recon Agent — Evaluation Metrics

How we measure whether the Recon Agent (`src/agents/recon.py`) is *good*, and —
critically for the research question — whether organizational context makes it
**measurably better** than the same agent without context.

Evaluation follows the agent loop, because each stage can fail independently:

```
READ (RAG context) → PLAN (probe selection) → PROBE → RECORD → PUBLISH (intel → RAG)
      [A]               [B]                   [C]      [D]          [E]
```

Metric classes:

- **[A] Context quality** — did the agent read the right organizational knowledge?
- **[B] Planning quality** — did context change probe choices, and correctly?
- **[C] Probing performance** — coverage, accuracy, cost, safety.
- **[D] Recording quality** — are findings structured and trustworthy?
- **[E] Memory loop** — do findings enrich the RAG and improve later runs?
- **[R] Research ablation metrics** — context-aware vs. context-blind runs.

All metrics must be computable from artifacts the agent already produces
(`Finding` records, RAG retrieval results, run timing) plus a **ground-truth
lab profile** (see §6) — nothing requires code changes to the agent itself.

---

## A. Context-quality metrics (READ stage)

Did `gather_context()` pull the right knowledge before acting?

| Metric | Definition | Formula | Good |
|---|---|---|---|
| A1. Context hit rate | % of targets for which RAG returns ≥1 chunk | `targets_with_context / targets` | 1.0 |
| A2. Correct-source rate | % of targets where the retrieved chunks include the *right* document (asset inventory for asset facts, topology for reachability) | `targets_with_correct_source / targets` | ≥ 0.9 |
| A3. Context precision | Of retrieved chunks, fraction actually relevant to the target (labeled) | `relevant_chunks / retrieved_chunks` | ≥ 0.6 |
| A4. Role correctness | % of targets where the extracted role line matches the asset's true role | `correct_roles / targets` | 1.0 |
| A5. Context latency | Wall time for `gather_context()` per target | measured (s) | < 1 s |

> A4 exists because we already found and fixed a bug here (all targets got
> "Payment gateway" as role). It is a regression metric: any future chunking or
> retrieval change must keep it at 1.0 on the testbed.

## B. Planning-quality metrics (PLAN stage)

Did context produce a *sensible* probe plan? Compared against what an expert
would probe given the same documents (expert plan = ground truth).

| Metric | Definition | Formula | Good |
|---|---|---|---|
| B1. Plan precision | Proportion of planned ports that the expert plan includes | `|planned ∩ expert| / |planned|` | ≥ 0.6 |
| B2. Plan recall | Proportion of expert ports the plan covers | `|planned ∩ expert| / |expert|` | ≥ 0.8 |
| B3. Plan order quality | Spearman correlation between planned port order and expert priority order | ρ over common ports | > 0 |
| B4. Context sensitivity | Does the plan change when context changes? Run planner with correct vs. shuffled/wrong context; plans should differ when hints differ | `Δ(planned ports)` | > 0 on hint-bearing cases |
| B5. Planning cost | Number of probes planned per target (fewer = cheaper, if recall holds) | `|planned|` | context-dependent |

B4 is the smallest unit of the research question: if plans are identical with
right vs. wrong context, the agent is not actually context-aware.

## C. Probing metrics (PROBE stage)

| Metric | Definition | Formula | Good |
|---|---|---|---|
| C1. Probe coverage | Planned probes actually executed | `executed / planned` | 1.0 |
| C2. Port-state accuracy | Open ports correctly reported open **and** closed ports correctly reported closed, vs. ground-truth lab state | `(TP+TN) / (TP+TN+FP+FN)` | ≥ 0.95 |
| C3. False-open rate | Closed ports reported open (worst error: invents attack surface) | `FP / (FP+TN_closed)` | 0 |
| C4. False-closed rate | Open ports missed (missed attack surface) | `FN / (FN+TP)` | ≤ 0.05 |
| C5. Probe wall time | Total probe phase duration per run/target | measured (s) | — |
| C6. Guardrail violations | Count of probes to non-approved ranges = **0, always** | count | **0 (hard)** |
| C7. Scan noise footprint | Distinct (target, port) connection attempts per run | count | minimize; fixed by plan |

C2–C4 need a live lab (§6). On the current testbed (targets down), the expected
result is 100% "filtered" — correct but uninformative, so these stay at 1.0
trivially; real signal requires the lab.

## D. Recording-quality metrics (RECORD stage)

| Metric | Definition | Formula | Good |
|---|---|---|---|
| D1. Finding completeness | Every executed probe yields exactly one finding | `findings / executed_probes` | 1.0 |
| D2. Schema validity | % of findings passing the `Finding` schema (categories, ISO timestamps) | `valid / total` | 1.0 |
| D3. Finding validity | % of findings that are true statements about the target (vs. ground truth) | `true_findings / total` | ≥ C2 |
| D4. Traceability | Every finding traceable to (run_id, target, probe) — no orphan findings | `traceable / total` | 1.0 |

## E. Memory-loop metrics (PUBLISH stage)

Does writing findings back into the RAG actually work and help?

| Metric | Definition | Formula | Good |
|---|---|---|---|
| E1. Publish success | % of runs whose intel report is written + indexed | `published / runs` | 1.0 |
| E2. Intel retrievability | For a standard probe query ("recon findings <target>"), intel chunks appear in top-k | hit rate over runs | ≥ 0.8 |
| E3. Self-knowledge gain | Run N answers "what did we find on X?" using intel chunks (not guessing) | % answered from intel | ≥ 0.8 |
| E4. Index growth control | Chunks added per run; total index size grows linearly, not explosively | chunks/run; total | bounded |
| E5. Duplicate/idempotency errors | Re-indexing same report creates duplicates or corrupts stats | count | 0 |
| E6. Cross-run improvement | Does run N+1 (with memory) plan better than run 1 (B2 recall)? | Δ B2 over runs | > 0 (eventually) |

E6 is the agent-learning claim of the architecture; it needs repeated runs
against a stable lab.

## R. Research ablation metrics (the core of the thesis)

Run the **same agent twice** per scenario: context-aware (full RAG) vs.
context-blind (RAG returns empty context — filters set to an impossible value,
or an empty-context flag). Compare on identical targets.

| Metric | Definition | Why it matters |
|---|---|---|
| R1. Plan difference rate | % of targets where probe plans differ between arms | Proves context changes behavior at all (paired with B4) |
| R2. Recall delta (ΔB2) | Context-aware plan recall − context-blind recall | Does context prevent probing dead paths (e.g., ports the topology says are firewalled) and include the right ones? |
| R3. Wasted-probe delta | Probes the topology says are unreachable, per arm | Context-aware should probe *fewer* impossible paths → cost + stealth benefit |
| R4. Risk-weighted yield | Σ over true findings of `criticality_weight(target)` — findings on high-criticality assets count more | The core "organizational impact" claim: same raw findings, better *prioritized* findings |
| R5. Narrative quality (LLM-judged) | Blind rating of the two agents' final reports on usefulness for a pentest lead | Captures the human-facing difference raw counts miss |
| R6. Consistency | Same-arm reruns produce the same plan (determinism) | Reproducibility requirement; `chunk_id` determinism should give 1.0 |

Statistical protocol: ≥ 10 scenarios × 2 arms, paired comparison per scenario
(same target set), report mean ± std and a paired test (Wilcoxon signed-rank)
for R2–R4. Determinism (R6) is asserted, not averaged.

---

## Current baseline (testbed, documented honestly)

The sample testbed (10.0.1.5 / 10.0.2.10 / 10.0.3.20) has **no live hosts**,
so:

| Metric | Current value | Meaning |
|---|---|---|
| A1 Context hit rate | 1.0 | RAG returns context for every target |
| A2 Correct-source rate | 1.0 | asset/topology/policy chunks retrieved per target |
| A4 Role correctness | 1.0 (after the fix) | roles extracted correctly per target |
| C1 Probe coverage | 1.0 | all planned probes executed |
| C2 Port-state accuracy | trivially 1.0 | all ports truly closed → "filtered" is correct |
| E1 Publish success | 1.0 | both live runs published + indexed |
| E2 Intel retrievability | 1.0 | intel chunks retrieved top-3 for the standard query |
| R6 Consistency | 1.0 | deterministic ids → same plan on rerun |

These prove the *pipeline* works. They do **not** yet prove the *agent is
useful* — that needs the ground-truth lab below.

## Ground-truth lab requirement

To turn trivial baselines into real measurements, stand up a small local lab:

1. 3–5 containers/VMs matching the sample inventory (payment gateway w/ 443,
   wiki w/ 80, HR DB w/ closed ports) — e.g. Docker on a private bridge network.
2. Freeze a **ground-truth profile** per host: listening ports, OS, role,
   reachability matrix (which source→target pairs are allowed).
3. Freeze an **expert probe plan** per host (what a pentester would check first,
   given the same org documents).
4. Re-run: C2–C4, B1–B3, R1–R5 become measurable; publish `results/recon_eval_<date>.json`
   with per-run metrics so the ablation study can aggregate them.

## Metric collection implementation notes

- All stage timings come from wrapping existing calls (gather_context /
  plan_probes / probe loop / publish_findings) — no agent rewrite needed.
- Findings already serialize via `findings_to_dicts()` → per-run JSON is the
  natural metrics source of truth.
- Ablation arm (context-blind) is a two-line change: call `run_recon` with
  retrieval forced to `[]` (a `context_enabled=False` flag), everything else
  identical — this is what makes R-metrics fair.
- Suggested tooling next step: `scripts/eval_recon.py` that runs N scenarios ×
  2 arms and emits the metrics table above into `results/`.
