"""
compare_models.py
-----------------
Side-by-side BEFORE (Review 1: SQuAD 1.1 DistilBERT) vs AFTER (Review 2: SQuAD 2.0
RoBERTa + no-answer threshold) on the hand-made sets. This is the slide that shows
the improvement.

Run:
    python compare_models.py
"""

from metrics import best_score_over_golds, compute_exact, compute_f1
from qa_engine import MODEL_NAME, V1_MODEL_NAME, QAEngine
from sample_data import SAMPLES, UNANSWERABLE


def show(r):
    return repr(r["answer"]) if r["has_answer"] else "[NO ANSWER]"


def main():
    old = QAEngine(V1_MODEL_NAME, threshold=0.0)
    new = QAEngine(MODEL_NAME)

    print("\n" + "=" * 78)
    print(f"{'QUESTION':<44}{'BEFORE (v1.1)':<18}{'AFTER (v2.0)'}")
    print("=" * 78)

    old_ok_unans = new_ok_unans = 0
    old_f1 = new_f1 = 0.0

    for ex in UNANSWERABLE:
        a = old.answer(ex["question"], ex["context"], allow_abstain=False)
        b = new.answer(ex["question"], ex["context"])
        old_ok_unans += int(not a["has_answer"])   # v1 can never abstain
        new_ok_unans += int(not b["has_answer"])
        mark = "ok" if not b["has_answer"] else "XX"
        print(f"{ex['question'][:42]:<44}{show(a)[:16]:<18}{show(b)[:26]} [{mark}]")

    print("-" * 78)
    for ex in SAMPLES:
        a = old.answer(ex["question"], ex["context"], allow_abstain=False)
        b = new.answer(ex["question"], ex["context"])
        old_f1 += best_score_over_golds(a["answer"], ex["gold_answers"], compute_f1)
        new_f1 += best_score_over_golds(b["answer"], ex["gold_answers"], compute_f1)

    nu, ns = len(UNANSWERABLE), len(SAMPLES)
    print("\nSUMMARY")
    print(f"  Unanswerable handled correctly : BEFORE {old_ok_unans}/{nu}   AFTER {new_ok_unans}/{nu}")
    print(f"  Answerable F1 (avg)            : BEFORE {100 * old_f1 / ns:.1f}%   AFTER {100 * new_f1 / ns:.1f}%")


if __name__ == "__main__":
    main()
