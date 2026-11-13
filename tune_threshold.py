"""
tune_threshold.py
-----------------
Finds the best "no answer" threshold on SQuAD 2.0 dev data and saves it to
threshold.json (qa_engine.py loads it automatically afterwards).

Lower threshold  -> abstains more often  (fewer wrong answers, may miss real ones)
Higher threshold -> answers more often   (may answer unanswerable questions)

Also supports per-language threshold tuning:
  --lang en  (default) : tune on SQuAD 2.0 → saves threshold_en.json
  --lang hi             : tune on Hindi hand-made sanity set → saves threshold_hi.json
                          WARNING: the Hindi set is tiny (~10 examples).
                          The result is unreliable and is printed with a warning.
                          Use it only as a starting point, not for publication.

Run:
    python tune_threshold.py            # English, 400 random dev examples
    python tune_threshold.py 800
    python tune_threshold.py --lang hi  # Hindi sanity set (tiny!)
Tune on a different sample than you report final numbers on (change --seed).
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _threshold_file(lang: str) -> str:
    """Return the threshold filename for a given language code."""
    if lang == "en":
        return os.path.join(HERE, "threshold.json")       # backward-compatible default
    return os.path.join(HERE, f"threshold_{lang}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", nargs="?", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lang", choices=["en", "hi"], default="en",
                    help="language to tune on: 'en' (SQuAD 2.0) or 'hi' (Hindi sanity set)")
    args = ap.parse_args()

    from datasets import load_dataset
    from metrics import best_score_over_golds, compute_f1
    from qa_engine import QAEngine

    threshold_file = _threshold_file(args.lang)

    engine = QAEngine(threshold=0.0)

    # ── Load tuning data ──────────────────────────────────────────────────────
    if args.lang == "en":
        squad = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=args.seed)
        squad = squad.select(range(min(args.n, len(squad))))
        examples = [{"question": ex["question"], "context": ex["context"],
                     "answers": ex["answers"]["text"]} for ex in squad]
        print(f"Scoring {len(examples)} English examples ...")

    else:  # Hindi
        from hindi_samples import HINDI_ANSWERABLE, HINDI_UNANSWERABLE
        examples = (
            [{"question": ex["question"], "context": ex["context"],
              "answers": ex["gold_answers"]} for ex in HINDI_ANSWERABLE]
            + [{"question": ex["question"], "context": ex["context"],
                "answers": ex["gold_answers"]} for ex in HINDI_UNANSWERABLE]
        )
        print(f"\n⚠️  WARNING: Hindi threshold tuning uses only {len(examples)} hand-made examples.")
        print("   The result is unreliable. Use as a rough starting point only.\n")
        print(f"Scoring {len(examples)} Hindi examples ...")

    # ── Collect score_diffs ───────────────────────────────────────────────────
    diffs, has_gold, span_f1 = [], [], []
    for i, ex in enumerate(examples, 1):
        r = engine.answer(ex["question"], ex["context"], allow_abstain=False)
        golds = ex["answers"]
        diffs.append(r["score_diff"])
        has_gold.append(bool(golds))
        span_f1.append(best_score_over_golds(r["candidate"], golds, compute_f1) if golds else 0.0)
        if i % 50 == 0:
            print(f"  {i}/{len(examples)}")

    diffs    = np.array(diffs)
    has_gold = np.array(has_gold)
    span_f1  = np.array(span_f1)

    def f1_at(thr):
        abstain = diffs > thr
        per_ex = np.where(has_gold,
                          np.where(abstain, 0.0, span_f1),
                          np.where(abstain, 1.0, 0.0))
        return 100.0 * per_ex.mean()

    grid = np.unique(np.concatenate([np.linspace(-10, 10, 81), diffs]))
    scores = [(f1_at(t), t) for t in grid]
    best_f1, best_thr = max(scores, key=lambda x: (x[0], -abs(x[1])))

    print("\nThreshold sweep (F1 overall):")
    for t in [-6, -4, -3, -2, -1, 0, 1, 2, 4]:
        print(f"  thr={t:>4}: F1={f1_at(t):.1f}")
    print(f"\nF1 at default threshold 0.0 : {f1_at(0.0):.1f}")
    print(f"Best threshold              : {best_thr:.3f}  ->  F1 {best_f1:.1f}")

    if args.lang == "hi":
        print(f"\n⚠️  WARNING: Hindi threshold tuned on only {len(examples)} examples.")
        print("   This is a rough guide — do not report this as a benchmark result.")

    with open(threshold_file, "w", encoding="utf-8") as f:
        json.dump({"threshold": round(float(best_thr), 3),
                   "tuned_on": len(examples),
                   "lang": args.lang,
                   "model": engine.model_name,
                   "f1": round(float(best_f1), 2)}, f, indent=2)
    print(f"Saved to {threshold_file}")


if __name__ == "__main__":
    main()
