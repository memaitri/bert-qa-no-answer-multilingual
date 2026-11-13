"""
evaluate_squad.py
-----------------
Evaluate on the REAL SQuAD 2.0 validation set (answerable + unanswerable questions),
reporting Overall / HasAns / NoAns EM and F1 like the official leaderboard.

Run:
    python evaluate_squad.py                  # 200 random examples (seed 42)
    python evaluate_squad.py 500              # 500 examples
    python evaluate_squad.py all              # full dev set (11,873 ex, slow on CPU)
    python evaluate_squad.py 300 --threshold -1.5
    python evaluate_squad.py 300 --no-abstain # force an answer every time (Review 1 behaviour)
"""

import argparse

from datasets import load_dataset

from metrics import aggregate, best_score_over_golds, compute_exact, compute_f1, print_report
from qa_engine import QAEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", default="200", help="number of examples or 'all'")
    ap.add_argument("--threshold", type=float, default=None, help="no-answer threshold (default: tuned/0.0)")
    ap.add_argument("--no-abstain", action="store_true", help="never abstain (SQuAD 1.1 behaviour)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    engine = QAEngine(threshold=args.threshold)

    print("\nLoading SQuAD 2.0 validation set ...")
    squad = load_dataset("rajpurkar/squad_v2", split="validation")
    if args.n.lower() != "all":
        n = min(int(args.n), len(squad))
        squad = squad.shuffle(seed=args.seed).select(range(n))  # random mix, not just the first article
    print(f"Evaluating on {len(squad)} examples (threshold={engine.threshold}, "
          f"abstain={'off' if args.no_abstain else 'on'})\n")

    records = []
    for i, ex in enumerate(squad, 1):
        r = engine.answer(ex["question"], ex["context"], allow_abstain=not args.no_abstain)
        golds = ex["answers"]["text"]
        em = best_score_over_golds(r["answer"], golds, compute_exact)
        f1 = best_score_over_golds(r["answer"], golds, compute_f1)
        records.append({"has_gold": bool(golds), "em": em, "f1": f1})
        if i % 50 == 0:
            a = aggregate(records)
            print(f"  {i}/{len(squad)}  EM={a['EM']:.1f}  F1={a['F1']:.1f}")

    print_report("SQuAD 2.0 VALIDATION RESULTS", aggregate(records))


if __name__ == "__main__":
    main()
