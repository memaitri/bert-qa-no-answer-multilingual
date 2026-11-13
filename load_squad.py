"""
load_squad.py
-------------
Explore the SQuAD 2.0 dataset (proves real-dataset usage to the jury).

Run:
    python load_squad.py           # print structure + a few answerable/unanswerable examples
    python load_squad.py --save    # also save 20 samples to squad_samples.json
"""

import json
import sys

from datasets import load_dataset


def main():
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    n_total = len(ds)
    n_unans = sum(1 for a in ds["answers"] if len(a["text"]) == 0)

    print("=" * 70)
    print("SQuAD 2.0 (validation)")
    print("=" * 70)
    print(f"Total questions : {n_total}")
    print(f"Answerable      : {n_total - n_unans}")
    print(f"Unanswerable    : {n_unans}  ({100 * n_unans / n_total:.1f}%)")
    print(f"Fields          : {list(ds.features.keys())}")

    shown_a = shown_u = 0
    for ex in ds:
        unans = len(ex["answers"]["text"]) == 0
        if (unans and shown_u >= 2) or (not unans and shown_a >= 2):
            continue
        shown_u += unans
        shown_a += not unans
        print("\n--- " + ("UNANSWERABLE" if unans else "ANSWERABLE") + " ---")
        print("Q      :", ex["question"])
        print("Context:", ex["context"][:150] + "...")
        print("Gold   :", ex["answers"]["text"] or "[no answer]")
        if shown_a >= 2 and shown_u >= 2:
            break

    if "--save" in sys.argv:
        rows = [dict(ds[i]) for i in range(20)]
        with open("squad_samples.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print("\nSaved 20 samples to squad_samples.json")


if __name__ == "__main__":
    main()
