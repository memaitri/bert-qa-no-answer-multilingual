"""
evaluate_multilingual.py
------------------------
Evaluate one or more QA models on English (SQuAD 2.0) and Hindi (XQuAD / MLQA),
and print a single comparison table.

WHY THIS COMPARISON?
  - deepset/roberta-base-squad2 was trained on English only → expected poor Hindi.
  - deepset/xlm-roberta-base-squad2 was trained on English SQuAD 2.0 but with
    an XLM-R backbone → it generalises to Hindi via zero-shot cross-lingual transfer.
  - Our fine-tuned checkpoint (./checkpoints/xlmr_squad2) was trained by us on
    SQuAD 2.0 with XLM-R; comparing it against the deepset XLM-R model shows
    whether our training matched or improved on the published baseline.

ABOUT HINDI ABSTENTION:
  XQuAD and MLQA Hindi splits are SQuAD 1.1 style — ALL questions are answerable.
  There are no unanswerable questions in those datasets.  Therefore:
  - We run with abstention ENABLED (default threshold) and DISABLED separately.
  - With abstention enabled, the model may refuse some questions → EM/F1 drops.
  - This shows the effect of the threshold on a purely answerable Hindi set.
  Real Hindi abstention can only be tested on hand-made data (hindi_sanity.py).

Run:
    python evaluate_multilingual.py
    python evaluate_multilingual.py --n 100 --seed 7
    python evaluate_multilingual.py --models deepset/roberta-base-squad2,deepset/xlm-roberta-base-squad2
    python evaluate_multilingual.py --models ./checkpoints/xlmr_squad2,deepset/xlm-roberta-base-squad2
"""

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = [
    "deepset/roberta-base-squad2",
    "deepset/xlm-roberta-base-squad2",
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_on_dataset(engine, examples, lang: str, allow_abstain: bool):
    """
    Run engine.answer() on a list of dicts with keys: question, context, answers.
    answers is a dict {"text": [...], "answer_start": [...]}.
    Returns aggregate dict from metrics.aggregate().
    """
    from metrics import (aggregate, best_score_over_golds,
                         compute_exact, compute_f1,
                         compute_exact_hindi, compute_f1_hindi)

    if lang == "hi":
        ex_fn, f1_fn = compute_exact_hindi, compute_f1_hindi
    else:
        ex_fn, f1_fn = compute_exact, compute_f1

    records = []
    for i, ex in enumerate(examples, 1):
        try:
            r = engine.answer(ex["question"], ex["context"],
                              allow_abstain=allow_abstain)
        except Exception:
            r = {"answer": "", "has_answer": False}

        golds = ex["answers"]["text"]
        em = best_score_over_golds(r["answer"], golds, ex_fn)
        f1 = best_score_over_golds(r["answer"], golds, f1_fn)
        records.append({"has_gold": bool(golds), "em": em, "f1": f1})

        if i % 50 == 0:
            print(f"    {i}/{len(examples)}")

    return aggregate(records)


def load_squad_en(n, seed):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": ex["question"], "context": ex["context"],
             "answers": ex["answers"]} for ex in ds]


def load_xquad_hi(n, seed):
    """Load XQuAD Hindi (all answerable, SQuAD 1.1 style)."""
    from datasets import load_dataset
    ds = load_dataset("xquad", "xquad.hi", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": ex["question"], "context": ex["context"],
             "answers": ex["answers"]} for ex in ds]


def load_mlqa_hi(n, seed):
    """Load MLQA Hindi (hi-hi config). Falls back gracefully if unavailable."""
    from datasets import load_dataset
    ds = load_dataset("mlqa", "mlqa-translate-test.hi", split="test",
                      trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": ex["question"], "context": ex["context"],
             "answers": ex["answers"]} for ex in ds]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model names or local paths")
    ap.add_argument("--n",    type=int, default=200,
                    help="examples per dataset (default 200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-mlqa", action="store_true",
                    help="skip MLQA (useful if the dataset is unavailable)")
    args = ap.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("Loading datasets ...")
    datasets = {}

    print("  SQuAD 2.0 English ...")
    try:
        datasets["en"] = load_squad_en(args.n, args.seed)
        print(f"    {len(datasets['en'])} examples")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        datasets["en"] = None

    print("  XQuAD Hindi ...")
    try:
        datasets["hi_xquad"] = load_xquad_hi(args.n, args.seed)
        print(f"    {len(datasets['hi_xquad'])} examples")
    except Exception as e:
        print(f"    ❌ XQuAD Hindi unavailable: {e}")
        datasets["hi_xquad"] = None

    if not args.skip_mlqa:
        print("  MLQA Hindi ...")
        try:
            datasets["hi_mlqa"] = load_mlqa_hi(args.n, args.seed)
            print(f"    {len(datasets['hi_mlqa'])} examples")
        except Exception as e:
            print(f"    ❌ MLQA Hindi unavailable (skipping): {e}")
            datasets["hi_mlqa"] = None
    else:
        datasets["hi_mlqa"] = None

    # ── Evaluate each model ───────────────────────────────────────────────────
    from qa_engine import QAEngine
    rows = []   # for CSV/MD output

    for model_name in model_list:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        try:
            engine = QAEngine(model_name=model_name)
        except Exception as e:
            print(f"  ❌ Could not load model: {e}")
            continue

        row = {"model": model_name}

        # English SQuAD 2.0
        if datasets["en"]:
            print(f"  English SQuAD 2.0 (n={len(datasets['en'])}) ...")
            agg = eval_on_dataset(engine, datasets["en"], lang="en", allow_abstain=True)
            row["en_em"]       = f"{agg['EM']:.1f}"
            row["en_f1"]       = f"{agg['F1']:.1f}"
            row["en_has_f1"]   = f"{agg['HasAns_F1']:.1f}"
            row["en_noans_em"] = f"{agg['NoAns_EM']:.1f}"
            print(f"    Overall EM={agg['EM']:.1f}  F1={agg['F1']:.1f}  "
                  f"HasAns_F1={agg['HasAns_F1']:.1f}  NoAns_EM={agg['NoAns_EM']:.1f}")
        else:
            row.update({"en_em": "--", "en_f1": "--",
                        "en_has_f1": "--", "en_noans_em": "--"})

        # XQuAD Hindi (abstain ON and OFF)
        for ds_key, ds_label in [("hi_xquad", "XQuAD Hindi"), ("hi_mlqa", "MLQA Hindi")]:
            if datasets[ds_key]:
                n_ex = len(datasets[ds_key])
                for abstain, suffix in [(False, "no_abstain"), (True, "abstain")]:
                    print(f"  {ds_label} (n={n_ex}, abstain={abstain}) ...")
                    agg = eval_on_dataset(engine, datasets[ds_key],
                                          lang="hi", allow_abstain=abstain)
                    k = f"{ds_key}_{suffix}"
                    row[f"{k}_em"] = f"{agg['EM']:.1f}"
                    row[f"{k}_f1"] = f"{agg['F1']:.1f}"
                    print(f"    EM={agg['EM']:.1f}  F1={agg['F1']:.1f}")
            else:
                for suffix in ["no_abstain", "abstain"]:
                    k = f"{ds_key}_{suffix}"
                    row[f"{k}_em"] = "--"
                    row[f"{k}_f1"] = "--"

        rows.append(row)

    if not rows:
        print("\nNo models evaluated successfully.")
        return

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("  MULTILINGUAL EVALUATION RESULTS")
    print(f"{'='*100}")
    headers = ["Model",
               "EN EM", "EN F1", "EN Has F1", "EN NoAns",
               "XQuAD-hi EM", "XQuAD-hi F1", "XQuAD-hi* EM", "XQuAD-hi* F1",
               "MLQA-hi EM",  "MLQA-hi F1",  "MLQA-hi* EM",  "MLQA-hi* F1"]
    keys    = ["model",
               "en_em", "en_f1", "en_has_f1", "en_noans_em",
               "hi_xquad_no_abstain_em", "hi_xquad_no_abstain_f1",
               "hi_xquad_abstain_em",    "hi_xquad_abstain_f1",
               "hi_mlqa_no_abstain_em",  "hi_mlqa_no_abstain_f1",
               "hi_mlqa_abstain_em",     "hi_mlqa_abstain_f1"]
    note = "  (* = with abstention enabled)"

    # Print header
    print(f"  {'Model':<45}", end="")
    for h in headers[1:]:
        print(f"  {h:>10}", end="")
    print()
    print("  " + "-"*45 + ("  " + "-"*10) * (len(headers)-1))

    for row in rows:
        print(f"  {row.get('model',''):<45}", end="")
        for k in keys[1:]:
            print(f"  {row.get(k,'--'):>10}", end="")
        print()

    print(f"\n  {note}")
    print(f"{'='*100}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(HERE, "results_multilingual.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = os.path.join(HERE, "results_multilingual.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Multilingual Evaluation Results\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["-"*max(5,len(h)+2) for h in headers]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k,"--")) for k in keys) + " |\n")
        f.write(f"\n{note}\n")
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    main()
