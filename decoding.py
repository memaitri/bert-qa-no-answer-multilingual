"""
decoding.py
-----------
Pure-numpy span decoding + the "no answer" decision rule.

Kept separate from the model code so it can be unit-tested without downloading
any model (see test_all.py --offline).

THE KEY IDEA (what changed in Review 2)
---------------------------------------
A SQuAD 2.0 model treats the [CLS] / <s> token (position 0) as a special
"there is no answer" pointer:

    null_score = start_logit[0] + end_logit[0]
    span_score = start_logit[s] + end_logit[e]      (best real span, s <= e)

    score_diff = null_score - best_span_score

If score_diff > threshold  ->  ABSTAIN ("answer not in passage")
else                       ->  return the best span

This is exactly the rule used in the official SQuAD 2.0 / HuggingFace
run_squad pipeline. The threshold can be tuned on dev data (tune_threshold.py).
"""

import math
import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sigmoid(x):
    # numerically safe
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def best_spans(start_logits, end_logits, context_mask, n_best=20, max_answer_len=30):
    """
    Return candidate spans as a list of (score, start_idx, end_idx), best first.

    Only tokens where context_mask is True can be part of an answer (so the
    question tokens, [CLS], [SEP] and padding are never extracted).
    Constraints: end >= start and span length <= max_answer_len tokens.
    """
    start_logits = np.asarray(start_logits, dtype=np.float64)
    end_logits = np.asarray(end_logits, dtype=np.float64)
    context_mask = np.asarray(context_mask, dtype=bool)

    start_idx = np.argsort(start_logits)[::-1][:n_best * 3]
    end_idx = np.argsort(end_logits)[::-1][:n_best * 3]

    # keep only context tokens, then take the top n_best of those
    start_idx = [int(i) for i in start_idx if context_mask[i]][:n_best]
    end_idx = [int(i) for i in end_idx if context_mask[i]][:n_best]

    cands = []
    for s in start_idx:
        for e in end_idx:
            if e < s or (e - s + 1) > max_answer_len:
                continue
            cands.append((float(start_logits[s] + end_logits[e]), s, e))
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands


def decide(null_score, best_span_score, threshold=0.0):
    """
    Returns (abstain: bool, score_diff: float).
    score_diff > 0 means the model prefers "no answer" over its best span.
    """
    diff = float(null_score) - float(best_span_score)
    return diff > threshold, diff


def answerable_probability(null_score, best_span_score):
    """
    Heuristic probability that the question IS answerable from the passage:
    sigmoid(best_span_score - null_score). 0.5 means the model is on the fence.
    """
    return sigmoid(float(best_span_score) - float(null_score))
