"""
calibration.py
--------------
Confidence calibration for the QA engine (Part B).

WHY CALIBRATION?
The heuristic confidence sigmoid(best_span - null) is NOT a calibrated
probability.  A value of 0.9 does not mean "this answer is correct 90% of the
time."  Calibration fits a post-hoc mapping from raw scores to true
probabilities, so the confidence can be trusted for downstream decisions (e.g.
"show the answer only when calibrated confidence > 0.7").

TWO CALIBRATORS
  1. Platt / temperature scaling  — fits a single sigmoid: P = σ(a·x + b).
     Good when the raw score is already roughly monotonic but needs re-scaling.
  2. Isotonic regression           — fits a piecewise-constant monotone function.
     More flexible, better when the raw score-probability relationship is
     non-linear, but needs more data to avoid overfitting.

WHAT WE CALL "CORRECTNESS"
For a binary calibration target we define:
  correct = 1  if the model's decision (answer vs abstain) was right:
                - answerable question AND model returned a non-empty answer, OR
                - unanswerable question AND model abstained.
  correct = 0  otherwise.

We use the heuristic answerable_prob as the raw confidence feature (x above).

Run:
    python calibration.py              # 400 examples, seed 99
    python calibration.py 800
    python calibration.py 400 --seed 13
"""

import argparse
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(HERE, "calibration_cache.json")
CALIBRATOR_FILE = os.path.join(HERE, "calibrator.pkl")

# ── ECE / Brier ───────────────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=10):
    """
    ECE: expected |accuracy - confidence| weighted by bin size.
    Lower is better.  A perfectly calibrated model has ECE = 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(probs))


def brier_score(probs, labels):
    """Mean squared error between predicted probability and 0/1 label."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


# ── Platt scaling ─────────────────────────────────────────────────────────────

class PlattScaler:
    """
    Fits P(correct | x) = σ(a·x + b) using logistic regression on the
    raw answerable_prob scores.
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, probs, labels):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(probs, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels, dtype=np.float64)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(X, y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])

    def predict(self, probs):
        x = np.asarray(probs, dtype=np.float64)
        z = self.a * x + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class IsotonicCalibrator:
    """Piecewise-constant monotone calibration using sklearn's IsotonicRegression."""
    def __init__(self):
        self._model = None

    def fit(self, probs, labels):
        from sklearn.isotonic import IsotonicRegression
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(np.asarray(probs), np.asarray(labels, dtype=float))

    def predict(self, probs):
        return self._model.predict(np.asarray(probs, dtype=float))


# ── Reliability diagram + risk-coverage curve ────────────────────────────────

def plot_reliability(raw_probs, labels, platt_probs, iso_probs, out_path):
    """
    Reliability diagram: fraction-correct vs mean-confidence per bin.
    A well-calibrated model's points fall on the diagonal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)

    for probs, label, color, ls in [
        (raw_probs,  "Raw sigmoid",        "#2196F3", "-"),
        (platt_probs,"Platt scaling",       "#4CAF50", "--"),
        (iso_probs,  "Isotonic regression", "#FF5722", ":"),
    ]:
        bin_acc, bin_conf = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (np.asarray(probs) >= lo) & (np.asarray(probs) < hi)
            if mask.sum() == 0:
                continue
            bin_acc.append(np.asarray(labels)[mask].mean())
            bin_conf.append(np.asarray(probs)[mask].mean())
        ax.plot(bin_conf, bin_acc, marker="o", label=label, color=color, linestyle=ls)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean confidence (predicted probability)", fontsize=11)
    ax.set_ylabel("Fraction correct", fontsize=11)
    ax.set_title("Reliability Diagram — QA Confidence Calibration", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def plot_risk_coverage(raw_probs, labels, platt_probs, iso_probs, out_path):
    """
    Risk-coverage curve: accuracy vs fraction of questions answered, sweeping
    the confidence threshold from low to high.

    Coverage = fraction of examples where confidence >= threshold.
    Accuracy = fraction correct among those answered.

    A good calibrator allows high accuracy at reasonable coverage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0, 1, 101)

    for probs, label, color, ls in [
        (raw_probs,  "Raw sigmoid",        "#2196F3", "-"),
        (platt_probs,"Platt scaling",       "#4CAF50", "--"),
        (iso_probs,  "Isotonic regression", "#FF5722", ":"),
    ]:
        p = np.asarray(probs)
        y = np.asarray(labels)
        coverages, accuracies = [], []
        for t in thresholds:
            mask = p >= t
            if mask.sum() == 0:
                break
            coverages.append(mask.mean())
            accuracies.append(y[mask].mean())
        ax.plot(coverages, accuracies, label=label, color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel("Coverage (fraction of questions answered)", fontsize=11)
    ax.set_ylabel("Accuracy (fraction correct among answered)", fontsize=11)
    ax.set_title("Risk-Coverage Curve — QA Confidence Calibration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", nargs="?", type=int, default=400,
                    help="number of SQuAD 2.0 dev examples (default 400)")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--test-split", type=float, default=0.3,
                    help="fraction held out for evaluation (default 0.3)")
    ap.add_argument("--force", action="store_true",
                    help="re-run model even if calibration_cache.json exists")
    args = ap.parse_args()

    # ── Collect or load cached scores ────────────────────────────────────────
    if os.path.exists(CACHE_FILE) and not args.force:
        print(f"Loading cached scores from {CACHE_FILE} ...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        raw_probs = np.array(cache["answerable_prob"])
        corrects  = np.array(cache["correct"])
        print(f"  {len(raw_probs)} examples loaded from cache.")
    else:
        from datasets import load_dataset
        from qa_engine import QAEngine

        print("Running model to collect calibration scores ...")
        engine = QAEngine()
        squad = load_dataset("rajpurkar/squad_v2", split="validation")
        squad = squad.shuffle(seed=args.seed).select(range(min(args.n, len(squad))))
        print(f"  {len(squad)} examples (seed={args.seed})")

        raw_probs, corrects = [], []
        for i, ex in enumerate(squad, 1):
            r = engine.answer(ex["question"], ex["context"])
            golds = ex["answers"]["text"]
            has_gold = bool(golds)
            # Correctness: right decision (answer↔abstain), not necessarily right span
            decision_correct = (has_gold and r["has_answer"]) or \
                               (not has_gold and not r["has_answer"])
            raw_probs.append(r["answerable_prob"])
            corrects.append(int(decision_correct))
            if i % 50 == 0:
                print(f"  {i}/{len(squad)}")

        raw_probs = np.array(raw_probs)
        corrects  = np.array(corrects)

        # Cache results so reruns are instant
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"answerable_prob": raw_probs.tolist(),
                       "correct": corrects.tolist(),
                       "n": len(corrects), "seed": args.seed}, f)
        print(f"Saved cache → {CACHE_FILE}")

    # ── Train / test split ────────────────────────────────────────────────────
    n = len(raw_probs)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - args.test_split)))
    train_idx, test_idx = idx[:split], idx[split:]

    p_train, y_train = raw_probs[train_idx], corrects[train_idx]
    p_test,  y_test  = raw_probs[test_idx],  corrects[test_idx]
    print(f"\nTrain: {len(train_idx)}   Test: {len(test_idx)}")

    # ── Fit calibrators ───────────────────────────────────────────────────────
    platt = PlattScaler()
    platt.fit(p_train, y_train)

    iso = IsotonicCalibrator()
    iso.fit(p_train, y_train)

    # Save the better calibrator (lower ECE on test) to CALIBRATOR_FILE
    p_platt_test = platt.predict(p_test)
    p_iso_test   = iso.predict(p_test)

    ece_raw   = expected_calibration_error(p_test,       y_test)
    ece_platt = expected_calibration_error(p_platt_test, y_test)
    ece_iso   = expected_calibration_error(p_iso_test,   y_test)

    bs_raw   = brier_score(p_test,       y_test)
    bs_platt = brier_score(p_platt_test, y_test)
    bs_iso   = brier_score(p_iso_test,   y_test)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  Calibration Results (held-out test set)")
    print("=" * 56)
    print(f"  {'Method':<26}  {'ECE':>7}   {'Brier':>7}")
    print(f"  {'-'*26}  {'-'*7}   {'-'*7}")
    print(f"  {'Raw sigmoid':<26}  {ece_raw:>7.4f}   {bs_raw:>7.4f}")
    print(f"  {'Platt scaling':<26}  {ece_platt:>7.4f}   {bs_platt:>7.4f}")
    print(f"  {'Isotonic regression':<26}  {ece_iso:>7.4f}   {bs_iso:>7.4f}")
    print("=" * 56)

    # Choose the calibrator with the lower ECE; save it
    chosen_name = "isotonic" if ece_iso <= ece_platt else "platt"
    chosen = iso if ece_iso <= ece_platt else platt
    with open(CALIBRATOR_FILE, "wb") as f:
        pickle.dump({"type": chosen_name, "calibrator": chosen}, f)
    print(f"\nSaved {chosen_name} calibrator → {CALIBRATOR_FILE}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    p_platt_all = platt.predict(raw_probs)
    p_iso_all   = iso.predict(raw_probs)

    plot_reliability(
        raw_probs, corrects, p_platt_all, p_iso_all,
        os.path.join(HERE, "reliability_diagram.png")
    )
    plot_risk_coverage(
        raw_probs, corrects, p_platt_all, p_iso_all,
        os.path.join(HERE, "risk_coverage.png")
    )


if __name__ == "__main__":
    main()
