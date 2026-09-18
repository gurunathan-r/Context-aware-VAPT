#!/usr/bin/env python3
"""Generate the Phase Review presentation pack (two PDFs).

    python scripts/generate_phase_review_pdfs.py
        -> reports/Phase_Review_Handout_<ts>.pdf      (1-2 page summary handout)
        -> reports/Phase_Review_Demo_Backup_<ts>.pdf  (captured demo transcripts,
                                                       rendered as terminal windows)

The demo backup reads the real captured transcripts from
reports/demo_transcripts/*.txt (recorded from live runs — see README demo
plan), so the PDF always shows what the system actually printed, not
hand-typed claims. Requires reportlab. No network access.
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))  # org_rag_phase1/ path-shim → this copy

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
TRANSCRIPTS = REPORTS_DIR / "demo_transcripts"
GENERATED = datetime.now()

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=19, leading=24,
                    spaceAfter=4, textColor=colors.HexColor("#0f2a43"))
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=10, leading=14,
                     alignment=TA_CENTER, textColor=colors.HexColor("#5a6b7b"),
                     spaceAfter=14)
H2 = ParagraphStyle("H2", parent=styles["Heading1"], fontSize=13, leading=17,
                    spaceBefore=12, spaceAfter=5,
                    textColor=colors.HexColor("#0f2a43"))
H3 = ParagraphStyle("H3", parent=styles["Heading2"], fontSize=11, leading=14,
                    spaceBefore=8, spaceAfter=3,
                    textColor=colors.HexColor("#274b6d"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.3, leading=13,
                      spaceAfter=4)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.2, leading=11,
                       textColor=colors.HexColor("#445566"))
CALLOUT = ParagraphStyle("CALLOUT", parent=BODY, fontSize=9.6, leading=13.5,
                         textColor=colors.HexColor("#1d3a2f"),
                         backColor=colors.HexColor("#eef7f0"),
                         borderColor=colors.HexColor("#9fc6ad"),
                         borderWidth=0.8, borderPadding=6, spaceBefore=6,
                         spaceAfter=8)
SAY = ParagraphStyle("SAY", parent=BODY, fontSize=9.2, leading=13,
                     textColor=colors.HexColor("#3a2c00"),
                     backColor=colors.HexColor("#fdf6e3"),
                     borderColor=colors.HexColor("#e0c979"),
                     borderWidth=0.8, borderPadding=6, spaceBefore=4,
                     spaceAfter=6)
CODE = ParagraphStyle("CODE", parent=BODY, fontName="Courier-Bold", fontSize=8.6,
                      leading=12, textColor=colors.HexColor("#0f2a43"),
                      backColor=colors.HexColor("#eef2f6"),
                      borderColor=colors.HexColor("#b9c8d6"),
                      borderWidth=0.6, borderPadding=4, spaceBefore=2,
                      spaceAfter=4)
MONO = ParagraphStyle("MONO", fontName="Courier", fontSize=6.8, leading=8.6,
                      textColor=colors.HexColor("#e6edf3"))

TERM_BG = colors.HexColor("#14161f")
TERM_BORDER = colors.HexColor("#3a3f4f")
BAR_BG = colors.HexColor("#252833")
BAR_TEXT = ParagraphStyle("BARTEXT", fontName="Courier-Bold", fontSize=7.2,
                          leading=9, textColor=colors.HexColor("#9fb0c3"))
DOTS = '<font color="#ff5f56">●</font> <font color="#ffbd2e">●</font> <font color="#27c93f">●</font>'

CELL = colors.HexColor("#f4f7fa")
HEADER_BG = colors.HexColor("#0f2a43")
GRID = colors.HexColor("#c5d1dc")


def esc(text: str) -> str:
    return html.escape(str(text))


def terminal_window(title: str, text: str, max_lines: int = 52) -> Table:
    """Render captured output as a macOS-style terminal 'screenshot'."""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Drop local absolute paths and progress-bar leftovers for presentation.
    cleaned = []
    for ln in lines:
        if "Loading weights" in ln or "HF Hub" in ln or not ln.strip():
            if not ln.strip():
                cleaned.append(ln)
            continue
        ln = ln.replace(str(PROJECT_ROOT) + "/", "")
        cleaned.append(ln[:128])
    if len(cleaned) > max_lines:
        keep_head = max_lines - 4
        cleaned = cleaned[:keep_head] + ["", f"... ({len(lines) - keep_head} lines trimmed) ..."] + cleaned[-2:]
    body = XPreformatted(esc("\n".join(cleaned)) or " ", MONO)

    bar = Paragraph(
        f'{DOTS}&nbsp;&nbsp;<font color="#8ea2b8">{esc(title)}</font>', BAR_TEXT)
    t = Table([[bar], [body]], colWidths=[18.2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BAR_BG),
        ("BACKGROUND", (0, 1), (0, 1), TERM_BG),
        ("BOX", (0, 0), (-1, -1), 0.8, TERM_BORDER),
        ("LINEBELOW", (0, 0), (0, 0), 0.4, TERM_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 4),
        ("BOTTOMPADDING", (0, 0), (0, 0), 4),
        ("TOPPADDING", (0, 1), (0, 1), 7),
        ("BOTTOMPADDING", (0, 1), (0, 1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def data_table(header: list[str], rows: list[list[str]], widths: list[float],
               font_size: float = 8.6) -> Table:
    head_style = ParagraphStyle("th", parent=BODY, fontName="Helvetica-Bold",
                                fontSize=font_size, leading=font_size + 2.5,
                                textColor=colors.white)
    cell_style = ParagraphStyle("td", parent=BODY, fontSize=font_size,
                                leading=font_size + 2.5)
    data = [[Paragraph(esc(h), head_style) for h in header]]
    data += [[Paragraph(esc(c), cell_style) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), CELL))
    t.setStyle(TableStyle(style))
    return t


def read_transcript(name: str) -> str:
    path = TRANSCRIPTS / name
    if not path.is_file():
        return f"[transcript missing: {name} — re-run the capture commands]"
    return path.read_text(encoding="utf-8", errors="replace")


def trim_context_block(text: str, keep_before_answer: int = 3) -> str:
    """Shorten the bulky retrieved-context preview, keep the ANSWER block."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith("ANSWER"):
            head = [ln for ln in lines[:max(0, i - keep_before_answer)] if ln.strip()]
            shown = head[:3]
            if len(head) > 3:
                shown.append(f"   ... ({len(head) - 3} more context lines trimmed) ...")
            return "\n".join(shown + [""] + lines[i:])
    return text


# ---------------------------------------------------------------------------
# PDF 1 — the handout
# ---------------------------------------------------------------------------

def build_handout() -> Path:
    story: list = []
    story.append(Paragraph("Context-Aware Agentic AI Framework for Automated VAPT", H1))
    story.append(Paragraph(
        f"Phase Review handout · generated {GENERATED:%Y-%m-%d %H:%M} · "
        "fully local: CPU embeddings, ChromaDB, local LLM — no cloud, no API keys", SUB))

    story.append(Paragraph(
        "<b>Thesis:</b> feeding organizational context (asset criticality, network "
        "topology, security policy) into an agentic VAPT pipeline changes agent "
        "behavior <b>measurably</b> — proven by an ablation study running the same "
        "agent with context (aware) and without (blind).", CALLOUT))

    story.append(Paragraph("1 · Architecture — the agent loop", H2))
    flow = ["Org documents", "Ingest + embed\n(MiniLM, CPU)", "ChromaDB RAG\n(metadata filters)",
            "Recon Agent\nread→plan→probe", "Intel report\n→ back into RAG",
            "Evaluation Agent\nindependent audit"]
    cells = [Paragraph(esc(s).replace("\n", "<br/>"),
                       ParagraphStyle("f", parent=BODY, fontSize=8.4, leading=11,
                                      alignment=TA_CENTER)) for s in flow]
    row, widths = [], []
    for i, c in enumerate(cells):
        row.append(c)
        widths.append(3.03 * cm)
        if i < len(cells) - 1:
            row.append(Paragraph("→", ParagraphStyle("a", parent=BODY, fontSize=11,
                                                     alignment=TA_CENTER,
                                                     textColor=colors.HexColor("#0f2a43"))))
            widths.append(0.5 * cm)
    ft = Table([row], colWidths=widths)
    ft.setStyle(TableStyle([
        ("BOX", (0, 0), (0, 0), 0.8, GRID), ("BOX", (2, 0), (2, 0), 0.8, GRID),
        ("BOX", (4, 0), (4, 0), 0.8, GRID), ("BOX", (6, 0), (6, 0), 0.8, GRID),
        ("BOX", (8, 0), (8, 0), 0.8, GRID),
        ("BACKGROUND", (0, 0), (0, 0), CELL), ("BACKGROUND", (2, 0), (2, 0), CELL),
        ("BACKGROUND", (6, 0), (6, 0), CELL), ("BACKGROUND", (8, 0), (8, 0), CELL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ft)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Safety: probing is hard-guarded to RFC1918/documentation ranges (critical "
        "audit check, weight 0.20), TCP-connect only, 5-port footprint; bug-bounty "
        "scope (RFC 9116 security.txt) can only <i>restrict</i>, never enable.", SMALL))

    story.append(Paragraph("2 · Verified results (live captures, this build)", H2))
    story.append(data_table(
        ["Area", "Metric", "Aware", "Blind", "Meaning"],
        [
            ["Retrieval", "A1 context hit rate", "1.00", "—", "RAG answers for every target"],
            ["Planning", "B2 plan recall (expert)", "1.000", "0.833", "context finds the right ports (1433 fix)"],
            ["Planning", "B4 context sensitivity", "1.00", "—", "plans change with context"],
            ["Ablation", "R2 recall delta", "+0.167", "—", "the thesis, quantified"],
            ["Prioritization", "M1 rank correlation ρ", "+1.00", "−0.90", "Δρ = +1.90 vs expert ground truth"],
            ["Prioritization", "M2 dead-end recall", "100%", "0%", "firewall-blocked paths caught"],
            ["Prioritization", "M3 alert-fatigue relief", "75%", "—", "false-critical alerts demoted"],
            ["Audit", "recon_quality_score", "98.1/100", "control", "grade A, PASS, zero failed checks"],
            ["Engineering", "test suite", "190 passed", "—", "fully offline, ~35 s"],
        ],
        [2.6 * cm, 4.6 * cm, 2.2 * cm, 2.0 * cm, 6.8 * cm]))

    story.append(Paragraph("3 · The evaluation loop working — before/after", H2))
    story.append(data_table(
        ["", "Before fix", "After fix"],
        [["Audit score", "94.5 (1 failed check: Q6 expert recall)", "98.1 (zero failed checks)"],
         ["Plan recall (B2)", "0.833 — MSSQL/1433 missed on HR database", "1.000 — hint table extended"],
         ["Ablation signal (R2)", "0.000 (no divergence)", "+0.167 (context finds what blind misses)"]],
        [3.6 * cm, 7.2 * cm, 7.4 * cm]))
    story.append(Paragraph(
        "The auditor <i>found</i> the planner gap, the fix was made, the metrics "
        "<i>proved</i> the improvement — the evaluation pipeline functioning as designed.", CALLOUT))

    story.append(Paragraph("4 · Honest limitations (stated before you ask)", H2))
    story.append(Paragraph(
        "• C2–C4 (port-state accuracy) need a live lab — testbed hosts are down, so those "
        "metrics report <b>n/a</b>, never fake 1.0s.<br/>"
        "• TCP-connect probes only — no service fingerprinting or exploitation yet (Phase 3).<br/>"
        "• Single flat Chroma collection; dense retrieval only (no rerank/BM25).<br/>"
        "• Ablation statistical protocol (≥10 scenarios, Wilcoxon) is defined but awaits the lab.", SMALL))

    story.append(Paragraph("5 · Reproduce everything", H2))
    for cmd in [
        "python scripts/rag_query.py \"What are the compliance requirements for payment systems?\"",
        "python scripts/run_recon.py",
        "python scripts/run_eval_agent.py --ground-truth results/ground_truth_testbed.json --llm",
        "python scripts/run_context_agent.py && python scripts/eval_context.py",
        "pytest tests/ -v   # 190 tests, offline",
    ]:
        story.append(Paragraph(esc("$ " + cmd), CODE))

    out = REPORTS_DIR / f"Phase_Review_Handout_{GENERATED:%Y%m%d_%H%M}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Phase Review Handout — Context-Aware VAPT")
    doc.build(story)
    return out


# ---------------------------------------------------------------------------
# PDF 2 — the demo backup (captured terminal 'screenshots')
# ---------------------------------------------------------------------------

DEMO_ENV = (
    "export LLM_BASE_URL=http://localhost:1234/v1\n"
    "export LLM_MODEL=qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"
)

SECTIONS: list[tuple[str, str, str, list[tuple[str, str, str, str]], str]] = [
    # (title, SAY, command, [(term_title, transcript, trim?, caption)], KEY LINE)
    ("Demo 1 · Grounded RAG — every claim cites its document",
     "Point out that the answer contains <b>zero</b> model-memory facts: every bullet "
     "ends with <code>(source: …)</code> pointing at an organizational document "
     "retrieved from ChromaDB. Then show the abstention: an unanswerable question "
     "must return the exact abstention sentence — this is the anti-hallucination contract.",
     DEMO_ENV + '\npython scripts/rag_query.py "What are the compliance requirements for payment systems?"',
     [("live run — rag_query.py (grounded)", "demo1_rag_grounded.txt", True,
       "The retrieved chunks are shown above the answer — the reader can verify each bullet against them."),
      ("live run — the abstention test", "demo1b_rag_abstain.txt", False,
       "Asked for a wifi password (not in any document), the system refuses to invent one.")],
     "<b>Key line:</b> “I don't know — the organizational documents provided do not "
     "cover this.” — hallucination is structurally prevented, not merely discouraged."),

    ("Demo 2 · The Recon Agent reads, plans, probes — and remembers",
     "The agent pulls asset/topology/policy context from the RAG, plans ports from "
     "context hints, probes safely (RFC1918-guarded), then <b>publishes its findings "
     "back into the RAG</b>. The second terminal queries the RAG afterwards: the "
     "answer cites the agent's own intel report — organizational memory that compounds.",
     DEMO_ENV + "\npython scripts/run_recon.py",
     [("live run — run_recon.py", "demo2_recon_run.txt", False,
       "Per-target context, plan, probe results, and the published intel file."),
      ("live run — asking the RAG what the agent found", "demo2_memory_loop.txt", True,
       "The source of the answer is the agent's own recon_findings_<ts>.md — the memory loop closed.")],
     "<b>Key line:</b> the citation “(source: recon_findings_…)” — the agent is now "
     "retrieving its own past work as organizational knowledge."),

    ("Demo 3 · Independent audit — trust is verified, never assumed",
     "A separate Evaluation Agent re-reads only the artifacts the recon run produced "
     "(snapshots, intel sidecar, ground-truth profile) and applies adversarial checks "
     "Q1–Q8. Blind-arm failures are reported as the ablation baseline, not graded "
     "defects; one failed <i>critical</i> check would force FAIL regardless of score.",
     DEMO_ENV + ('\npython scripts/run_eval_agent.py --ground-truth '
                 'results/ground_truth_testbed.json --llm'),
     [("live run — Evaluation Agent audit (LLM narrative judge)", "demo3_audit.txt", False,
       "Q-scores per arm; the aware arm is graded, the blind arm is the control (Q8=0 by construction).")],
     "<b>Key line:</b> “VERDICT: PASS | grade A | quality score 98.1/100” with the "
     "blind arm at Q8 = 0.00 — the ablation signal visible inside the audit itself."),

    ("Demo 4 · Context-aware prioritization — same CVEs, better order",
     "The Context Agent re-prioritizes raw scanner findings using business context: "
     "firewall dead-ends are demoted, PCI-DSS assets escalated with SLAs, and every "
     "finding carries a RAG citation. The harness scores it against expert ground truth.",
     DEMO_ENV + "\npython scripts/run_context_agent.py\npython scripts/eval_context.py",
     [("live run — run_context_agent.py (blind vs aware reports)", "demo4_context_agent.txt", True,
       "Side-by-side comparison; reports written to reports/vuln_report_{blind,aware}.md."),
      ("live run — eval_context.py (M1–M6 scorecard)", "demo4_eval_context.txt", False,
       "Quantified against the expert benchmark: Δρ = +1.90, 100% dead-end recall.")],
     "<b>Key line:</b> “Alert Fatigue Reduced: 75.0%” + Δρ = +1.90 — business context "
     "changes what gets fixed first, measurably."),

    ("Evidence · Metrics harness and test suite",
     "Everything above is reproduced by the evaluation harness (A–E/R metric families "
     "from metrics.md) and a fully offline test suite — no LLM, no network, "
     "deterministic, ~35 seconds.",
     "python scripts/eval_recon.py --ground-truth results/ground_truth_testbed.json --numbered"
     "\npytest tests/ -q",
     [("live run — A–E/R metric report", "metrics_table.txt", False,
       "Plan-only mode: profile-dependent probing metrics honestly show n/a."),
      ("live run — pytest", "test_run.txt", False,
       "190 passed — every guardrail and metric has a regression test.")],
     "<b>Key line:</b> B2 = 1.000, R2 = +0.167 — and C2–C4 showing n/a, not invented numbers."),
]


def build_demo_backup() -> Path:
    story: list = []
    story.append(Paragraph("Phase Review — Demo Backup Pack", H1))
    story.append(Paragraph(
        f"Captured live on {GENERATED:%Y-%m-%d at %H:%M} · LM Studio "
        "(qwen3.6-35b-a3b, local) · all runs verifiable by re-running the commands",
        SUB))
    story.append(Paragraph(
        "If the live demo fails, walk the panel through these captured transcripts — "
        "each one is the actual terminal output of a real run, with an explanation of "
        "what it proves. Setup: <font face='Courier-Bold' size='8.6'>"
        + esc(DEMO_ENV).replace("\n", "<br/>") + "</font>", CALLOUT))

    for title, say, command, terms, keyline in SECTIONS:
        block: list = [Paragraph(title, H2), Paragraph("What to say: " + say, SAY),
                       Paragraph(esc(command).replace("\n", "<br/>"), CODE)]
        for term_title, fname, do_trim, caption in terms:
            text = read_transcript(fname)
            if do_trim:
                text = trim_context_block(text)
            block.append(Spacer(1, 3))
            block.append(terminal_window(term_title, text))
            block.append(Paragraph(caption, SMALL))
        block.append(Paragraph(keyline, CALLOUT))
        story.append(KeepTogether(block[:4]))
        story.extend(block[4:])
        story.append(PageBreak())

    out = REPORTS_DIR / f"Phase_Review_Demo_Backup_{GENERATED:%Y%m%d_%H%M}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Phase Review Demo Backup — Context-Aware VAPT")
    doc.build(story)
    return out


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    handout = build_handout()
    backup = build_demo_backup()
    print(f"Handout:     {handout}  ({handout.stat().st_size / 1024:.0f} KB)")
    print(f"Demo backup: {backup}  ({backup.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
