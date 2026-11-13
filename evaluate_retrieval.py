"""
evaluate_retrieval.py
---------------------
Evaluate the retriever + reader pipeline end-to-end on SQuAD 2.0 (Part A).

HOW IT WORKS
  1. Take N SQuAD 2.0 dev examples. Collect all unique contexts as documents.
  2. Write them to temporary .txt files so the Retriever can load them.
  3. For each question, retrieve top-k chunks and check whether the gold answer
     appears in any of the top-1 / top-3 / top-5 chunks (Recall@k).
  4. Run DocQA.ask() end-to-end and compute EM / F1 for answerable questions
     and abstain rate for unanswerable ones.

Run:
    python evaluate_retrieval.py              # 100 examples, top-5
    python evaluate_retrieval.py 200
    python evaluate_retrieval.py 100 --seed 7
"""

import argparse
import os
import tempfile

from datasets import load_dataset

from doc_qa import DocQA
from metrics import aggregate, best_score_over_golds, compute_exact, compute_f1


def answer_in_chunk(gold_answers, chunk_text):
    """Return True if any gold answer string appears (case-insensitive) in chunk_text."""
    ct = chunk_text.lower()
    return any(g.lower() in ct for g in gold_answers)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", nargs="?", type=int, default=100,
                    help="number of SQuAD 2.0 dev examples (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading SQuAD 2.0 validation split ...")
    squad = load_dataset("rajpurkar/squad_v2", split="validation")
    squad = squad.shuffle(seed=args.seed).select(range(min(args.n, len(squad))))
    print(f"  {len(squad)} examples (seed={args.seed})")

    # ── Write unique contexts to temp .txt files ──────────────────────────────
    # Each SQuAD context becomes one document.  We deduplicate to avoid
    # inflating Recall@k by having the same passage appear many times.
    unique_contexts = list({ex["context"]: ex["context"] for ex in squad}.values())
    tmpdir = tempfile.mkdtemp(prefix="squad_docs_")
    doc_paths = []
    for i, ctx in enumerate(unique_contexts):
        path = os.path.join(tmpdir, f"doc_{i:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(ctx)
        doc_paths.append(path)
    print(f"  {len(doc_paths)} unique contexts written to {tmpdir}")

    # ── Build DocQA and load documents ───────────────────────────────────────
    dqa = DocQA()
    print(f"Loading and embedding {len(doc_paths)} documents ...")
    dqa.load(doc_paths)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    recall_at = {1: 0, 3: 0, 5: 0}
    records = []   # for EM / F1
    n_answerable = 0
    n_unanswerable = 0
    n_abstained_on_unanswerable = 0

    print(f"\nEvaluating {len(squad)} questions (top_k={args.top_k}) ...")

    for i, ex in enumerate(squad, 1):
        question = ex["question"]
        golds = ex["answers"]["text"]
        has_gold = bool(golds)

        # ── Retrieval Recall@k ────────────────────────────────────────────────
        retrieved = dqa.retriever.retrieve(question, doc_paths, top_k=max(5, args.top_k))

        for k in [1, 3, 5]:
            top_chunks = [item["chunk"].text for item in retrieved[:k]]
            if has_gold and any(answer_in_chunk(golds, ct) for ct in top_chunks):
                recall_at[k] += 1

        # ── End-to-end QA ─────────────────────────────────────────────────────
        result = dqa.ask(question, top_k=args.top_k)

        if has_gold:
            n_answerable += 1
            em = best_score_over_golds(result["answer"], golds, compute_exact)
            f1 = best_score_over_golds(result["answer"], golds, compute_f1)
            records.append({"has_gold": True, "em": em, "f1": f1})
        else:
            n_unanswerable += 1
            if not result["has_answer"]:
                n_abstained_on_unanswerable += 1
            # For aggregate, unanswerable is correct iff we abstained
            records.append({
                "has_gold": False,
                "em": 1.0 if not result["has_answer"] else 0.0,
                "f1": 1.0 if not result["has_answer"] else 0.0,
            })

        if i % 20 == 0 or i == len(squad):
            print(f"  {i}/{len(squad)}")

    # ── Print results ─────────────────────────────────────────────────────────
    n_ans_total = max(1, n_answerable)  # guard div-by-zero

    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION  (Recall@k)")
    print("=" * 60)
    for k in [1, 3, 5]:
        pct = 100.0 * recall_at[k] / n_ans_total
        print(f"  Recall@{k}: {pct:.1f}%  ({recall_at[k]}/{n_answerable} answerable questions)")
    print("=" * 60)

    from metrics import aggregate, print_report
    agg = aggregate(records)
    print_report(
        f"END-TO-END DOC QA  (n={len(squad)}, top_k={args.top_k})",
        agg
    )

    if n_unanswerable > 0:
        abstain_rate = 100.0 * n_abstained_on_unanswerable / n_unanswerable
        print(f"Unanswerable abstain rate: {abstain_rate:.1f}%  "
              f"({n_abstained_on_unanswerable}/{n_unanswerable})")

    # ── Cleanup temp files ────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
