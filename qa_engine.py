"""
qa_engine.py
------------
Extractive Question-Answering engine with "no answer" detection.

REVIEW 2 UPGRADE
================
Review 1 used a SQuAD 1.1 model (distilbert-base-cased-distilled-squad). It was
trained without unanswerable questions, so when the passage did not contain the
answer it still returned a random span ("Who directed Inception?" -> "Leonardo
DiCaprio").

This version uses a model fine-tuned on SQuAD 2.0 (deepset/roberta-base-squad2),
which has an extra "no answer" option, and adds a tunable decision rule:

    null_score = start_logit[CLS] + end_logit[CLS]
    best_score = best real span score (start_logit[s] + end_logit[e])
    abstain    = (null_score - best_score) > threshold

Other fixes over the Review 1 engine
------------------------------------
* Answer tokens are restricted to CONTEXT tokens using the tokenizer's
  sequence_ids (before, the "offset != (0,0)" check also let question tokens in).
* Long passages: overlapping windows; best span across windows, and the null
  score is the minimum over windows (official SQuAD 2.0 recipe).
* Uses raw logits (as in the official recipe) instead of summed softmax values.
* Returns the model's best guess even when abstaining (for transparency) and a
  heuristic answerability probability.

How BERT-style extractive QA locates a span (Jury Q1)
-----------------------------------------------------
1. Input is [CLS] question [SEP] context [SEP].
2. The encoder gives a contextual vector per token.
3. A linear QA head produces one start logit and one end logit per token.
4. The answer is the span (s <= e) with the highest start[s] + end[e].
5. Token 0 ([CLS]) is the "no answer" pointer in SQuAD 2.0 training.
"""

import json
import os

import numpy as np

from decoding import (answerable_probability, best_spans, decide, softmax)

# Default: SQuAD 2.0 model. Override with env var, e.g.
#   QA_MODEL=deepset/minilm-uncased-squad2 python demo_app.py   (smaller / faster)
MODEL_NAME = os.environ.get("QA_MODEL", "deepset/roberta-base-squad2")
V1_MODEL_NAME = "distilbert-base-cased-distilled-squad"  # Review 1 model (for comparison)

THRESHOLD_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold.json")
CALIBRATOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrator.pkl")

# Per-language threshold files written by tune_threshold.py --lang <code>
_LANG_THRESHOLD_FILES = {
    "en": THRESHOLD_FILE,
    "hi": os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_hi.json"),
}


def load_default_threshold(lang: str = "en") -> float:
    """
    Use the tuned threshold for the given language if present, else 0.0.
    Falls back to threshold.json (English) if the language-specific file is absent.
    """
    path = _LANG_THRESHOLD_FILES.get(lang, THRESHOLD_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return 0.0


def load_calibrator():
    """
    Load the post-hoc calibrator saved by calibration.py, if it exists.
    Returns the calibrator object (has a .predict(probs) method), or None.
    Keeping this in a standalone function lets it be called lazily once.
    """
    try:
        import pickle
        with open(CALIBRATOR_FILE, "rb") as f:
            bundle = pickle.load(f)
        return bundle["calibrator"]
    except Exception:
        return None


class QAEngine:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None,
                 threshold: float = None, lang: str = "en"):
        """
        lang: language code for threshold loading ("en" or "hi").
              The QA logic is identical regardless of language — the XLM-R
              tokenizer uses the same CLS-at-index-0 convention as RoBERTa.
              We use sequence_ids() (not token_type_ids) to identify context
              tokens, which works for XLM-R, RoBERTa, BERT, and DistilBERT.
        """
        # heavy imports are local so decoding/metrics stay importable without torch
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.lang = lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = load_default_threshold(lang) if threshold is None else threshold

        print(f"Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Load calibrator once at startup; None if calibration.py hasn't been run yet.
        self._calibrator = load_calibrator()
        cal_status = "loaded" if self._calibrator is not None else "not found (run calibration.py)"
        print(f"Model loaded. No-answer threshold = {self.threshold}  |  calibrator: {cal_status}")

    def answer(self, question: str, context: str, threshold: float = None,
               allow_abstain: bool = True, max_length: int = 384, stride: int = 128,
               n_best: int = 20, max_answer_len: int = 30, batch_size: int = 8):
        """
        Returns a dict:
            answer         extracted text, or "" if the model abstains
            has_answer     False if the model decided the passage has no answer
            confidence     confidence in the *decision* made (0-1)
            answerable_prob heuristic P(question is answerable)
            score_diff     null_score - best_span_score (>threshold => abstain)
            threshold      threshold that was used
            candidate      best span text even if abstained (what v1 would have said)
            top_candidates up to 3 distinct (text, score) alternatives
            start, end     character offsets of the returned span (None if abstained)
        """
        if not question or not question.strip():
            raise ValueError("Question is empty.")
        if not context or not context.strip():
            raise ValueError("Context is empty.")

        torch = self._torch
        thr = self.threshold if threshold is None else threshold

        enc = self.tokenizer(
            question,
            context,
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        n_windows = len(enc["input_ids"])
        keys = [k for k in self.tokenizer.model_input_names if k in enc]

        null_scores = []
        cands = []  # (score, text, char_start, char_end, span_prob)

        for b in range(0, n_windows, batch_size):
            idx = list(range(b, min(b + batch_size, n_windows)))
            batch = {k: torch.tensor([enc[k][i] for i in idx], device=self.device) for k in keys}
            with torch.no_grad():
                out = self.model(**batch)
            start_all = out.start_logits.detach().cpu().numpy()
            end_all = out.end_logits.detach().cpu().numpy()

            for j, i in enumerate(idx):
                sl, el = start_all[j], end_all[j]
                seq_ids = enc.sequence_ids(i)
                mask = np.array([s == 1 for s in seq_ids], dtype=bool)  # context tokens only
                offsets = enc["offset_mapping"][i]

                null_scores.append(float(sl[0] + el[0]))
                p_start, p_end = softmax(sl), softmax(el)

                for score, s, e in best_spans(sl, el, mask, n_best, max_answer_len):
                    cs, ce = offsets[s][0], offsets[e][1]
                    text = context[cs:ce].strip()
                    if text:
                        cands.append((score, text, cs, ce, float(p_start[s] * p_end[e])))

        cands.sort(key=lambda c: c[0], reverse=True)
        null_score = min(null_scores) if null_scores else 0.0

        if not cands:  # nothing extractable at all
            return {"answer": "", "has_answer": False, "confidence": 1.0, "answerable_prob": 0.0,
                    "score_diff": float("inf"), "threshold": thr, "candidate": "",
                    "top_candidates": [], "start": None, "end": None}

        best_score, best_text, cs, ce, span_prob = cands[0]
        abstain, diff = decide(null_score, best_score, thr)
        if not allow_abstain:
            abstain = False
        p_ans = answerable_probability(null_score, best_score)

        # distinct alternatives for display
        seen, top = set(), []
        for sc, tx, _, _, _ in cands:
            key = tx.lower()
            if key not in seen:
                seen.add(key)
                top.append((tx, round(sc, 3)))
            if len(top) == 3:
                break

        if allow_abstain:
            confidence = (1.0 - p_ans) if abstain else p_ans
        else:  # v1-style: confidence is just the span probability
            confidence = span_prob

        # Optional calibrated probability — present only when calibrator.pkl exists.
        # The calibrator maps answerable_prob -> P(decision is correct), which is a
        # genuine probability unlike the raw sigmoid heuristic.
        calibrated_prob = None
        if self._calibrator is not None:
            try:
                cal_arr = self._calibrator.predict([float(p_ans)])
                calibrated_prob = round(float(cal_arr[0]), 3)
            except Exception:
                calibrated_prob = None

        result = {
            "answer": "" if abstain else best_text,
            "has_answer": not abstain,
            "confidence": round(float(confidence), 3),
            "answerable_prob": round(float(p_ans), 3),
            "score_diff": round(float(diff), 3),
            "threshold": thr,
            "candidate": best_text,
            "top_candidates": top,
            "start": None if abstain else cs,
            "end": None if abstain else ce,
        }
        if calibrated_prob is not None:
            result["calibrated_prob"] = calibrated_prob
        return result


if __name__ == "__main__":
    engine = QAEngine()
    ctx = "Python was created by Guido van Rossum and first released in 1991."
    for q in ["Who created Python?", "Who created Java?"]:
        r = engine.answer(q, ctx)
        print(q, "->", repr(r["answer"]) if r["has_answer"] else "[NO ANSWER]", r)
