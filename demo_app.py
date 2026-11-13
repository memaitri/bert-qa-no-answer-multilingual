"""
demo_app.py
-----------
Live Gradio demo — three tabs with a polished two-column layout.

  Tab 1 — Passage QA        : inputs left, results right; answer card +
                               confidence meter; collapsible "Why?" accordion.
  Tab 2 — Document QA       : file upload with status, answer card, source
                               chunk, collapsible chunk table.
  Tab 3 — Multilingual demo : side-by-side pass/fail cards for RoBERTa vs
                               XLM-R on Hindi examples.

UI rules kept here:
  - No model/decoding logic changed.  Handler functions only format output.
  - Custom CSS lives in ui_style.css.
  - Gradio 4.x+ (currently 6.3.0) — no new heavy deps.

Run:
    python demo_app.py
Then open http://127.0.0.1:7860
"""

import os
import gradio as gr

from qa_engine import QAEngine, load_default_threshold, MODEL_NAME
from sample_data import FAILURE_CASE, SAMPLES, UNANSWERABLE
from hindi_samples import HINDI_ANSWERABLE, HINDI_UNANSWERABLE

# ── Load CSS ──────────────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "ui_style.css")
with open(_CSS_PATH, "r", encoding="utf-8") as _f:
    _CSS = _f.read()

# ── Device badge ──────────────────────────────────────────────────────────────
def _device_label() -> str:
    try:
        import torch
        return "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "CPU"

_DEVICE = _device_label()

# ── Model registry ────────────────────────────────────────────────────────────
_AVAILABLE_MODELS = {
    "RoBERTa (English only)":             "deepset/roberta-base-squad2",
    "XLM-RoBERTa deepset (multilingual)": "deepset/xlm-roberta-base-squad2",
}
_LOCAL_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "xlmr_squad2")
if os.path.isdir(_LOCAL_CKPT):
    _AVAILABLE_MODELS["XLM-RoBERTa (our fine-tuned)"] = _LOCAL_CKPT

_DEFAULT_MODEL_LABEL = "RoBERTa (English only)"

_engine_cache: dict = {}

def _get_engine(model_label: str, lang: str) -> QAEngine:
    model_name = _AVAILABLE_MODELS.get(model_label, MODEL_NAME)
    key = (model_name, lang)
    if key not in _engine_cache:
        _engine_cache[key] = QAEngine(model_name=model_name, lang=lang)
    return _engine_cache[key]

_default_engine: QAEngine = None
def _default_eng() -> QAEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = QAEngine()
    return _default_engine

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _answer_card(answer: str, has_answer: bool) -> str:
    """Render a green 'Answer' or amber 'No answer' result card."""
    if has_answer:
        return (
            '<div class="answer-card has-answer">'
            '<div class="answer-label">✅ Answer</div>'
            f'<div class="answer-text">{answer}</div>'
            '</div>'
        )
    return (
        '<div class="answer-card no-answer">'
        '<div class="answer-label">🚫 No answer found in the passage</div>'
        f'<div class="answer-text">{answer or "The model abstained."}</div>'
        '</div>'
    )


def _confidence_bar(prob: float, label: str) -> str:
    """Render a thin progress-bar confidence meter."""
    pct = int(round(prob * 100))
    return (
        '<div class="conf-wrap">'
        f'<div class="conf-label">{label}: <strong>{pct}%</strong></div>'
        '<div class="conf-bar-bg">'
        f'<div class="conf-bar-fill" style="width:{pct}%"></div>'
        '</div></div>'
    )


def _empty_state(msg: str = "Ask a question to see the answer here.") -> str:
    return f'<div class="empty-state">{msg}</div>'


def _header_html(active_model: str) -> str:
    return (
        '<div id="qa-header">'
        "<h1>Extractive QA with No-Answer Detection</h1>"
        "<p>Answers are extracted directly from the passage — the model never "
        "generates new words, and <em>abstains</em> when the answer is not there.</p>"
        f'<span class="badge">Model: {active_model}</span>'
        f'<span class="badge">Device: {_DEVICE}</span>'
        '<span class="badge">SQuAD 2.0</span>'
        "</div>"
    )


# ── Tab 1: Passage QA — handlers ──────────────────────────────────────────────

def ask(context, question, model_label, lang, threshold):
    """Run the reader and return (highlight, card_html, conf_html, details_md)."""
    # ── Validation ────────────────────────────────────────────────────────────
    if not (context or "").strip():
        return (
            None,
            '<div class="validation-msg">⚠️ Please enter a passage.</div>',
            "",
            "",
        )
    if not (question or "").strip():
        return (
            None,
            '<div class="validation-msg">⚠️ Please enter a question.</div>',
            "",
            "",
        )

    try:
        engine = _get_engine(model_label, lang)
        r = engine.answer(question, context, threshold=float(threshold))
    except Exception as e:
        return None, f'<div class="validation-msg">Error: {e}</div>', "", ""

    # ── Highlighted passage (character offsets safe for Unicode/Devanagari) ───
    if r["has_answer"]:
        s, e = r["start"], r["end"]
        highlighted = [(context[:s], None), (context[s:e], "answer"), (context[e:], None)]
    else:
        highlighted = [(context, None)]

    # ── Answer card ───────────────────────────────────────────────────────────
    card = _answer_card(r["answer"] if r["has_answer"] else "", r["has_answer"])

    # ── Confidence bar: prefer calibrated_prob if available ───────────────────
    if "calibrated_prob" in r:
        conf_html = _confidence_bar(r["calibrated_prob"],
                                    "Calibrated confidence (P correct)")
    else:
        conf_html = _confidence_bar(r["answerable_prob"],
                                    "Heuristic P(answerable)  — not calibrated")

    card = card + conf_html   # attach below the card

    # ── "Why this answer?" technical details ──────────────────────────────────
    alts_lines = [f"`{t}`  (score {sc})" for t, sc in r["top_candidates"]]
    rejected = ""
    if not r["has_answer"] and r["candidate"]:
        rejected = f"\n**Best span rejected:** `{r['candidate']}`"
    cal_line = ""
    if "calibrated_prob" in r:
        cal_line = f"\n- Calibrated P(correct): **{r['calibrated_prob']}**"

    details_md = (
        f"- score\\_diff: **{r['score_diff']}**  (abstain when > threshold **{r['threshold']}**)\n"
        f"- P(answerable): **{r['answerable_prob']}**  |  confidence: **{r['confidence']}**"
        + cal_line
        + ("\n\n**Top candidate spans:**\n" + "\n".join(alts_lines) if alts_lines else "")
        + rejected
    )

    return highlighted, card, details_md, ""


def on_lang_change(lang):
    thr = load_default_threshold(lang)
    return gr.update(value=thr)


# ── Tab 2: Document QA — handlers ─────────────────────────────────────────────

_doc_qa = None
_loaded_paths = []
_loaded_chunk_count = 0

def _get_doc_qa():
    global _doc_qa
    if _doc_qa is None:
        from doc_qa import DocQA
        _doc_qa = DocQA(qa_engine=_default_eng())
    return _doc_qa


def upload_docs(files, top_k, threshold):
    global _loaded_paths, _loaded_chunk_count
    if not files:
        return '<div class="doc-status">📂 No files uploaded yet.</div>'
    try:
        paths = [f.name for f in files]
        dqa = _get_doc_qa()
        dqa.load(paths)
        _loaded_paths = paths
        # Count chunks from retriever cache
        key = tuple(sorted(os.path.abspath(p) for p in paths))
        chunks, _ = dqa.retriever._cache.get(key, ([], None))
        _loaded_chunk_count = len(chunks)
        names = ", ".join(os.path.basename(p) for p in paths)
        return (
            f'<div class="doc-status">✅ Loaded <strong>{len(paths)}</strong> file(s) '
            f'→ <strong>{_loaded_chunk_count}</strong> chunks indexed.<br>'
            f'<span style="font-size:.8rem;opacity:.8">{names}</span></div>'
        )
    except Exception as e:
        return f'<div class="doc-status" style="color:#DC2626">❌ Error: {e}</div>'


def doc_ask(files, question, top_k, threshold):
    global _loaded_paths
    if not (question or "").strip():
        return (
            '<div class="validation-msg">⚠️ Please enter a question.</div>',
            "",
            "",
        )
    if files:
        paths = [f.name for f in files]
        if sorted(paths) != sorted(_loaded_paths):
            upload_docs(files, top_k, threshold)

    if not _loaded_paths:
        return (
            '<div class="validation-msg">⚠️ Please upload at least one document first.</div>',
            "",
            "",
        )
    try:
        dqa = _get_doc_qa()
        result = dqa.ask(question, top_k=int(top_k), threshold=float(threshold))
    except Exception as e:
        return f'<div class="validation-msg">Error: {e}</div>', "", ""

    # ── Answer card ───────────────────────────────────────────────────────────
    if result["has_answer"]:
        src = os.path.basename(result["source_file"])
        score = result["retrieval_score"]
        card = _answer_card(result["answer"], True)
        card += (
            f'<div style="font-size:.82rem;margin-top:6px;color:#6b7280">'
            f'📄 Source: <strong>{src}</strong> &nbsp;|&nbsp; '
            f'retrieval score: <strong>{score:.3f}</strong></div>'
        )
        if "answerable_prob" in result:
            card += _confidence_bar(result["answerable_prob"],
                                    "Heuristic P(answerable)")
    else:
        card = _answer_card("", False)

    # ── Highlighted source chunk (plain text with inline bold for answer) ─────
    chunk = result["source_chunk"]
    chunk_md = ""
    if chunk is not None:
        ct = chunk.text
        ans = result["answer"]
        if ans and ans in ct:
            s = ct.index(ans)
            chunk_md = (
                f"**Source:** `{os.path.basename(chunk.source)}`  "
                f"(chars {chunk.char_start}–{chunk.char_end})\n\n"
                f"{ct[:s]}**[{ans}]**{ct[s+len(ans):]}"
            )
        else:
            chunk_md = (
                f"**Source:** `{os.path.basename(chunk.source)}`\n\n{ct}"
            )

    # ── Chunk table (Markdown) ────────────────────────────────────────────────
    rows = ["| Rank | Score | Answer | diff | File |",
            "|------|-------|--------|------|------|"]
    for d in result["chunk_decisions"]:
        c = d["chunk"]
        ans_cell = ("✅ " + d["answer"][:45]) if d["has_answer"] else "🚫 abstain"
        rows.append(
            f"| {d['rank']} | {d['retrieval_score']:.3f} | {ans_cell} "
            f"| {d['score_diff']:.2f} | {os.path.basename(c.source)} |"
        )
    table_md = "\n".join(rows) if result["chunk_decisions"] else "_No chunks retrieved._"

    return card, chunk_md, table_md


# ── Tab 3: Multilingual — handler ────────────────────────────────────────────

def run_both(context, question):
    if not (context or "").strip() or not (question or "").strip():
        empty = _empty_state("Enter a Hindi passage and question, then click Run.")
        return empty, empty

    results = {}
    for label in ["RoBERTa (English only)", "XLM-RoBERTa deepset (multilingual)"]:
        try:
            eng = _get_engine(label, "hi")
            r = eng.answer(question, context)
            results[label] = r
        except Exception as ex:
            results[label] = {"has_answer": False, "answer": "",
                              "candidate": str(ex), "answerable_prob": 0.0}

    def _ml_card(label, r):
        short = label.split("(")[0].strip()
        if r["has_answer"]:
            inner = (
                f'<div class="ml-model-name">{short}</div>'
                f'<div style="color:var(--green);font-weight:600;font-size:1.05rem">'
                f'✅ {r["answer"]}</div>'
                f'<div style="font-size:.78rem;color:#6b7280;margin-top:4px">'
                f'P(ans)={r.get("answerable_prob",0):.2f}</div>'
            )
            cls = "pass"
        else:
            cand = r.get("candidate", "")
            inner = (
                f'<div class="ml-model-name">{short}</div>'
                f'<div style="color:var(--amber);font-weight:600">🚫 Abstained</div>'
                + (f'<div style="font-size:.82rem;color:#6b7280;margin-top:4px">'
                   f'Would have said: <em>{cand}</em></div>' if cand else "")
            )
            cls = "fail"
        return f'<div class="ml-card {cls}">{inner}</div>'

    r1 = results["RoBERTa (English only)"]
    r2 = results["XLM-RoBERTa deepset (multilingual)"]
    return _ml_card("RoBERTa (English only)", r1), \
           _ml_card("XLM-RoBERTa deepset (multilingual)", r2)


# ── Example data ──────────────────────────────────────────────────────────────
_en_examples = (
    [[s["context"], s["question"]] for s in SAMPLES[:4]]
    + [[FAILURE_CASE["context"], FAILURE_CASE["question"]]]
    + [[u["context"], u["question"]] for u in UNANSWERABLE[1:3]]
)
_hi_examples = (
    [[ex["context"], ex["question"]] for ex in HINDI_ANSWERABLE[:4]]
    + [[ex["context"], ex["question"]] for ex in HINDI_UNANSWERABLE[:3]]
)

# ── Build the app ─────────────────────────────────────────────────────────────

_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Noto Sans Devanagari", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    button_primary_background_fill="#4F6EF7",
    button_primary_background_fill_hover="#3B55D4",
    button_primary_text_color="white",
    block_radius="12px",
    input_radius="10px",
    button_large_radius="10px",
)

with gr.Blocks(
    title="Extractive QA — SQuAD 2.0 + Hindi",
) as demo:

    # ── Compact header ─────────────────────────────────────────────────────────
    gr.HTML(_header_html(_DEFAULT_MODEL_LABEL))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Passage QA
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("📖 Passage QA"):
        with gr.Row(elem_classes="two-col-row"):

            # ── LEFT column: inputs ──────────────────────────────────────────
            with gr.Column(scale=5):
                context_box = gr.Textbox(
                    label="Context / Passage",
                    lines=8,
                    placeholder="Paste your passage here (English or Hindi)…",
                )
                question_box = gr.Textbox(
                    label="Question",
                    placeholder="What do you want to know?",
                )

                with gr.Row():
                    btn = gr.Button("Get Answer", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Accordion("⚙️ Advanced settings", open=False):
                    with gr.Row():
                        model_dd = gr.Dropdown(
                            choices=list(_AVAILABLE_MODELS.keys()),
                            value=_DEFAULT_MODEL_LABEL,
                            label="Model",
                        )
                        lang_dd = gr.Dropdown(
                            choices=[("English", "en"), ("Hindi (हिंदी)", "hi")],
                            value="en",
                            label="Language",
                        )
                    threshold = gr.Slider(
                        -8, 8, value=0.0, step=0.25,
                        label="No-answer threshold  "
                              "(↓ abstains more  |  ↑ answers more)",
                    )

                with gr.Accordion("💡 English examples", open=False):
                    gr.Examples(
                        examples=_en_examples,
                        inputs=[context_box, question_box],
                        label="Click a row to load it",
                    )
                with gr.Accordion("💡 Hindi examples (हिंदी उदाहरण)", open=False):
                    gr.Examples(
                        examples=_hi_examples,
                        inputs=[context_box, question_box],
                        label="Click a row to load it",
                    )

            # ── RIGHT column: results ────────────────────────────────────────
            with gr.Column(scale=5):
                result_card = gr.HTML(_empty_state())

                highlight_box = gr.HighlightedText(
                    label="Passage (answer highlighted)",
                    show_legend=False,
                    visible=True,
                )

                with gr.Accordion("🔍 Why this answer?", open=False):
                    details_box = gr.Markdown(
                        value="_Run a query to see technical details._"
                    )
                # hidden detail string — kept for internal use, not shown
                _hidden = gr.Textbox(visible=False)

        # ── Wire up ──────────────────────────────────────────────────────────
        lang_dd.change(on_lang_change, inputs=[lang_dd], outputs=[threshold])

        btn.click(
            ask,
            inputs=[context_box, question_box, model_dd, lang_dd, threshold],
            outputs=[highlight_box, result_card, details_box, _hidden],
        )
        clear_btn.click(
            lambda: ("", "", _empty_state(), [], "_Run a query to see technical details._"),
            outputs=[context_box, question_box, result_card, highlight_box, details_box],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Document QA
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("📄 Document QA"):
        with gr.Row(elem_classes="two-col-row"):

            # ── LEFT: upload + question ──────────────────────────────────────
            with gr.Column(scale=5):
                file_upload = gr.File(
                    label="Upload documents (.txt or .pdf)",
                    file_count="multiple",
                    file_types=[".txt", ".pdf"],
                )
                doc_status = gr.HTML(
                    '<div class="doc-status">📂 No files loaded yet.</div>'
                )
                doc_question = gr.Textbox(
                    label="Question",
                    placeholder="Ask something about your documents…",
                )
                with gr.Row():
                    doc_btn   = gr.Button("Ask", variant="primary", scale=3)
                    doc_clear = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Accordion("⚙️ Advanced settings", open=False):
                    with gr.Row():
                        top_k_slider  = gr.Slider(1, 10, value=5, step=1,
                                                  label="Top-k chunks to retrieve")
                        doc_threshold = gr.Slider(-8, 8, value=0.0, step=0.25,
                                                  label="No-answer threshold")

            # ── RIGHT: results ───────────────────────────────────────────────
            with gr.Column(scale=5):
                doc_result_card = gr.HTML(_empty_state())

                with gr.Accordion("📌 Source chunk", open=True):
                    doc_chunk = gr.Markdown(
                        value="_The relevant chunk will appear here._"
                    )

                with gr.Accordion("📋 All retrieved chunks", open=False):
                    doc_table = gr.Markdown(
                        value="_Retrieved chunks with scores will appear here._"
                    )

        # ── Wire up ──────────────────────────────────────────────────────────
        file_upload.change(
            upload_docs,
            inputs=[file_upload, top_k_slider, doc_threshold],
            outputs=[doc_status],
        )
        doc_btn.click(
            doc_ask,
            inputs=[file_upload, doc_question, top_k_slider, doc_threshold],
            outputs=[doc_result_card, doc_chunk, doc_table],
        )
        doc_clear.click(
            lambda: (
                _empty_state(),
                "_The relevant chunk will appear here._",
                "_Retrieved chunks with scores will appear here._",
                "",
            ),
            outputs=[doc_result_card, doc_chunk, doc_table, doc_question],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Multilingual demo
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("🌐 Multilingual demo"):
        gr.Markdown(
            "## Zero-shot Hindi QA\n"
            "XLM-R is fine-tuned on **English SQuAD 2.0 only** — no Hindi training "
            "data. Click an example to see the English-only RoBERTa struggle and "
            "XLM-R succeed via cross-lingual transfer."
        )

        with gr.Row(elem_classes="two-col-row"):
            with gr.Column(scale=5):
                ml_context  = gr.Textbox(
                    label="Hindi Passage (context)",
                    lines=6,
                    placeholder="हिंदी अनुच्छेद यहाँ पेस्ट करें…",
                )
                ml_question = gr.Textbox(
                    label="Hindi Question (हिंदी प्रश्न)",
                    placeholder="प्रश्न लिखें…",
                )
                ml_btn = gr.Button("Run both models", variant="primary")

                with gr.Accordion("💡 Hindi examples (हिंदी उदाहरण)", open=True):
                    gr.Examples(
                        examples=[[ex["context"], ex["question"]]
                                  for ex in HINDI_ANSWERABLE[:5]],
                        inputs=[ml_context, ml_question],
                        label="Click a row to load it",
                    )

            with gr.Column(scale=5):
                gr.Markdown("### Results")
                with gr.Row():
                    roberta_card = gr.HTML(
                        _empty_state("RoBERTa result will appear here.")
                    )
                    xlmr_card = gr.HTML(
                        _empty_state("XLM-R result will appear here.")
                    )
                gr.Markdown(
                    "<small>🔵 RoBERTa was trained on English only → "
                    "expected to fail on Hindi.<br>"
                    "🟣 XLM-R uses a multilingual encoder → "
                    "zero-shot Hindi transfer.</small>"
                )

        ml_btn.click(
            run_both,
            inputs=[ml_context, ml_question],
            outputs=[roberta_card, xlmr_card],
        )

    # ── Footer ──────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="qa-footer">'
        "Extractive QA — answers come from the passage only; "
        "the model <em>abstains</em> when the answer is not there. &nbsp;|&nbsp; "
        "Topic 180 · SQuAD 2.0 · deepset/roberta-base-squad2"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(theme=_theme, css=_CSS)
