"""
evaluate_samples.py   (was evaluate.py in Review 1 -- renamed so it can never
shadow the HuggingFace `evaluate` package)
-----------------------------------------------------------------------------
Quick evaluation on the hand-made sets: answerable questions (EM/F1) and
unanswerable questions (did the model correctly abstain?).

Run:
    python evaluate_samples.py
"""

from metrics import aggregate, best_score_over_golds, compute_exact, compute_f1, print_report
from qa_engine import QAEngine
from sample_data import SAMPLES, UNANSWERABLE


def run_evaluation(engine=None):
    engine = engine or QAEngine()
    records = []

    print("\n" + "=" * 70)
    print(f"ANSWERABLE QUESTIONS ({len(SAMPLES)})")
    print("=" * 70)
    for i, ex in enumerate(SAMPLES, 1):
        r = engine.answer(ex["question"], ex["context"])
        em = best_score_over_golds(r["answer"], ex["gold_answers"], compute_exact)
        f1 = best_score_over_golds(r["answer"], ex["gold_answers"], compute_f1)
        records.append({"has_gold": True, "em": em, "f1": f1})
        shown = repr(r["answer"]) if r["has_answer"] else "[NO ANSWER]"
        print(f"[{i}] Q: {ex['question']}")
        print(f"    Predicted: {shown}  (conf={r['confidence']}, diff={r['score_diff']})")
        print(f"    Gold     : {ex['gold_answers']}   EM={em}  F1={f1:.2f}")

    print("\n" + "=" * 70)
    print(f"UNANSWERABLE QUESTIONS ({len(UNANSWERABLE)}) -- correct behaviour = abstain")
    print("=" * 70)
    for i, ex in enumerate(UNANSWERABLE, 1):
        r = engine.answer(ex["question"], ex["context"])
        ok = not r["has_answer"]
        records.append({"has_gold": False, "em": float(ok), "f1": float(ok)})
        verdict = "OK  abstained" if ok else f"WRONG answered {r['answer']!r}"
        print(f"[{i}] Q: {ex['question']}")
        print(f"    {verdict}  (would-be guess={r['candidate']!r}, conf={r['confidence']}, diff={r['score_diff']})")

    print_report("SAMPLE-SET RESULTS (SQuAD 2.0 style)", aggregate(records))


if __name__ == "__main__":
    run_evaluation()
