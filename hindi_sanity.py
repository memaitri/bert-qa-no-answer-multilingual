"""
hindi_sanity.py
---------------
Sanity check: run one or more models on the hand-written Hindi QA pairs
(hindi_samples.py) and report:
  - Per-example verdict (correct / wrong / abstained)
  - Accuracy on answerable questions (is the gold answer substring of predicted?)
  - Abstain rate on unanswerable questions

IMPORTANT: This is a ~10+10 sample SANITY CHECK, not a benchmark.
Use evaluate_multilingual.py with XQuAD for real numbers.

Run:
    python hindi_sanity.py
    python hindi_sanity.py --models deepset/xlm-roberta-base-squad2,./checkpoints/xlmr_squad2
"""

import argparse

from hindi_samples import HINDI_ANSWERABLE, HINDI_UNANSWERABLE
from metrics import normalize_hindi


def run_sanity(model_name: str, threshold: float = None):
    from qa_engine import QAEngine
    engine = QAEngine(model_name=model_name, threshold=threshold or 0.0)
    thr = engine.threshold

    print(f"\n{'='*65}")
    print(f"  Model: {model_name}  (threshold={thr})")
    print(f"{'='*65}")

    # ── Answerable ────────────────────────────────────────────────────────────
    print(f"\n  ANSWERABLE ({len(HINDI_ANSWERABLE)} questions)")
    print(f"  {'-'*60}")
    n_correct = 0
    for ex in HINDI_ANSWERABLE:
        r = engine.answer(ex["question"], ex["context"])
        pred = normalize_hindi(r["answer"])
        golds = [normalize_hindi(g) for g in ex["gold_answers"]]
        # Soft match: gold appears as substring of pred or pred in gold
        correct = any(g in pred or pred in g for g in golds if g)
        if correct:
            n_correct += 1
        status = "✅" if correct else ("🚫 abstained" if not r["has_answer"] else "❌ wrong")
        print(f"  Q: {ex['question'][:50]}")
        print(f"     Pred: {r['answer'][:60] or '[abstained]'}  {status}")

    # ── Unanswerable ──────────────────────────────────────────────────────────
    print(f"\n  UNANSWERABLE ({len(HINDI_UNANSWERABLE)} questions)")
    print(f"  {'-'*60}")
    n_abstained = 0
    for ex in HINDI_UNANSWERABLE:
        r = engine.answer(ex["question"], ex["context"])
        abstained = not r["has_answer"]
        if abstained:
            n_abstained += 1
        status = "✅ abstained" if abstained else f"❌ answered: {r['answer'][:40]!r}"
        print(f"  Q: {ex['question'][:50]}")
        print(f"     {status}  (diff={r['score_diff']:.2f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*45}")
    print(f"  Answerable   accuracy : {n_correct}/{len(HINDI_ANSWERABLE)}")
    print(f"  Unanswerable abstained: {n_abstained}/{len(HINDI_UNANSWERABLE)}")
    print(f"  {'─'*45}")

    return {
        "model":          model_name,
        "ans_correct":    n_correct,
        "ans_total":      len(HINDI_ANSWERABLE),
        "unans_abstained":n_abstained,
        "unans_total":    len(HINDI_UNANSWERABLE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="deepset/roberta-base-squad2",
                    help="comma-separated model names or paths")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    summaries = []
    for m in model_list:
        try:
            s = run_sanity(m, threshold=args.threshold)
            summaries.append(s)
        except Exception as e:
            print(f"❌ Could not load {m}: {e}")

    if summaries:
        print(f"\n{'='*65}")
        print("  SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Model':<45}  {'Ans Acc':>8}  {'Abstain':>8}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*8}")
        for s in summaries:
            ans_pct = 100 * s["ans_correct"]  / max(1, s["ans_total"])
            abs_pct = 100 * s["unans_abstained"] / max(1, s["unans_total"])
            print(f"  {s['model']:<45}  {ans_pct:>7.0f}%  {abs_pct:>7.0f}%")


if __name__ == "__main__":
    main()
