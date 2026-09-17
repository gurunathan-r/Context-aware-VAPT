# Plans: Organizational Context Agent & Evaluation Framework

## 1. Executive Summary & Objective

**Project Goal**: Demonstrate that integrating organizational context (asset criticality, business unit ownership, network reachability/topology, regulatory compliance) via local RAG substantially improves VAPT report quality compared to conventional context-blind (raw CVSS) vulnerability reporting.

**Review Rubric**: *"Prove that your context agent makes a difference in report (with suitable evaluation metrics)."*

Since scanning and exploitation agents are deferred to a later phase, we simulate a realistic vulnerability scanner dataset (`data/simulated_vulns.json`) covering the target testbed. We then implement:
1. **The Organizational Context Agent (`src/agents/context_agent.py`)** to analyze vulnerabilities through the organizational RAG knowledge base.
2. **Comparative Reporting (`scripts/run_context_agent.py`)** showing side-by-side Context-Blind vs. Context-Aware reports.
3. **The Evaluation Harness (`src/eval_context.py` & `scripts/eval_context.py`)** with mathematical metrics proving ranking superiority, dead-end discovery, and alert fatigue reduction against expert ground truth.

---

## 2. Core Shortcomings of Context-Blind VAPT (The Baseline)

Conventional vulnerability assessment tools (Nessus, Qualys, OpenVAS) rank vulnerabilities strictly by their CVSS v3.1 base score. This leads to two critical flaws:
- **Dead-End Wasted Effort**: Scanning tools do not understand internal firewall policies (e.g. `FW-014`). A CVSS 9.8 vulnerability on an internal port of a segmented server is flagged as Critical, even though ingress is blocked by perimeter firewalls. Pentesters and security teams waste hours chasing unexploitable paths.
- **Business Misalignment & False Urgency**: An RCE vulnerability (CVSS 9.8) on an internal test/documentation wiki (low criticality, internal only, data backed up in git) outranks an auth bypass or SQL injection (CVSS 7.5) on an internet-facing payment gateway (processing cardholder data, subject to PCI-DSS mandatory 30-day remediation SLAs).

---

## 3. Architecture & Data Flow

```
                               ┌─────────────────────────────┐
                               │  Simulated Scanner Findings │
                               │  (data/simulated_vulns.json)│
                               └──────────────┬──────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
            [Context-Blind Arm]                            [Context-Aware Arm]
       (Sort purely by raw CVSS)                     (src/agents/context_agent.py)
                       │                                             │
                       │                                             ├── 1. RAG Retrieval (Asset, Topology, Policy)
                       │                                             ├── 2. Dead-End / Reachability Check (FW-014)
                       │                                             ├── 3. Business Criticality & Compliance Multiplier
                       │                                             └── 4. Grounded Citation Extraction
                       ▼                                             ▼
            [Baseline Vuln Report]                        [Context-Aware Vuln Report]
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                                 [Evaluation Harness (eval_context.py)]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [Quantitative Evaluation Metrics]                 [Qualitative Report Diff]
         - Spearman Rank Correlation (ρ)                   - Identified dead ends
         - Dead-End Detection Rate                         - Business unit owners
         - Alert Fatigue Reduction (FUR)                   - Compliance SLA enforcement
         - Compliance SLA Alignment                        - Document citations
```

---

## 4. Evaluation Metrics Framework

To satisfy the review rubric ("prove that your context agent makes a difference with suitable evaluation metrics"), we implement five formal metrics:

| Metric | Code | Definition & Formula | Context-Blind Expected | Context-Aware Expected |
|---|---|---|---|---|
| **Ranking Alignment with Expert Business Priority** | **M1** | Spearman's rank correlation $\rho$ and Kendall's $\tau$ between the generated report rank and the expert ground-truth priority rank. | Low or Negative ($\rho \le 0.20$) | High ($\rho \ge 0.85 - 1.00$) |
| **Dead-End Detection Rate** | **M2** | Precision and Recall of identifying unexploitable/firewalled vulnerabilities (e.g., firewall rule `FW-014` blocking internal reachability). | Recall = 0.0 (Flags all as active) | Recall = 1.0 (Identifies all dead ends) |
| **False Urgency Reduction (FUR)** | **M3** | $\frac{\text{Demoted High/Critical False Urgencies}}{\text{Total Raw High/Critical}} \times 100\%$ — measures reduction in alert fatigue for isolated/low-impact assets. | 0% | $\ge 40\%$ |
| **Compliance SLA Alignment Rate** | **M4** | Proportion of regulatory-bound vulnerabilities (PCI-DSS, GDPR) elevated to match mandatory remediation SLAs (e.g. 30 days). | $\le 50\%$ | 100% |
| **Context Grounding / Citation Rate** | **M5** | Fraction of prioritized findings containing explicit, auditable document citations `(source: <file>)`. | 0% | 100% |

---

## 5. Implementation Roadmap

### Step 1: Simulated Vulnerability Dataset & Ground Truth
- `data/simulated_vulns.json`: 5 realistic vulnerabilities across the 3 testbed hosts (`10.0.1.5`, `10.0.2.10`, `10.0.3.20`), covering high-CVSS internal assets, dead-end firewall rules (`FW-014`), and PCI-DSS/GDPR regulated assets.
- `results/vuln_ground_truth.json`: Expert ground truth mapping each vulnerability to true business priority, dead-end status, compliance impact, and SLA.

### Step 2: Org Context Agent (`src/agents/context_agent.py`)
- Query ChromaDB RAG for asset inventory, network topology, and security policy context.
- Reachability & Dead-End detector: inspects target port and ingress route against firewall rules in `network_topology.txt`.
- Context-Adjusted Business Risk scoring engine:
  $$Risk_{context} = CVSS \times W_{criticality} \times M_{compliance} \times M_{exposure} \times M_{reachability}$$
- Report generators for both arms:
  - Context-Blind Report: Standard CVSS-ordered list.
  - Context-Aware Report: Business risk-ordered list with dead-end callouts, compliance warnings, and document citations.

### Step 3: Evaluation Engine (`src/eval_context.py` & `scripts/eval_context.py`)
- Implements M1–M5 metrics calculations.
- Generates `results/context_eval_<timestamp>.json` and prints an executive summary table.

### Step 4: CLI Demonstration Runner (`scripts/run_context_agent.py`)
- Runs both arms side-by-side.
- Prints comparative table and highlights differences (re-ranked vulnerabilities, filtered dead ends, compliance flags).

### Step 5: Unit & Integration Tests (`tests/test_context_agent.py`, `tests/test_eval_context.py`)
- Comprehensive offline test suite verifying all metrics, ranking logic, and edge cases.

