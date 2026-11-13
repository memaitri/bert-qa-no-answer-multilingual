"""
metrics.py
----------
Official SQuAD Exact Match / F1, plus SQuAD 2.0 style HasAns / NoAns breakdown.
Also provides language-aware normalization for Hindi / Devanagari text.

For an unanswerable question the gold answer list is empty and the model is
only correct if it predicts the empty string "" (i.e. it abstains).
"""

import re
import string
import unicodedata
import collections


def normalize_hindi(s: str) -> str:
    """
    Normalize a Hindi / Devanagari answer string for EM / F1 comparison.

    WHY different from normalize_answer?
    - Unicode NFC normalization is needed because Devanagari vowel marks can be
      encoded in multiple ways (precomposed vs decomposed); without it two
      visually identical strings may compare as unequal.
    - The English article removal ("a", "an", "the") must NOT be applied to
      Hindi, because those words do not exist and the regex would corrupt
      short Hindi words that happen to contain those letter sequences.
    - The Devanagari danda (।, U+0964) and double danda (॥, U+0965) are
      sentence-end punctuation analogous to the English period; strip them.
    - Devanagari digits (०–९) are left as-is; we do not map them to ASCII
      digits because gold answers in XQuAD/MLQA use native-script digits.
    - For Latin-script parts (transliterations inside Hindi text), we lowercase.
    """
    s = unicodedata.normalize("NFC", s)             # canonical Unicode form
    # Strip Devanagari danda and double-danda
    s = s.replace("।", "").replace("॥", "")
    # Strip standard ASCII punctuation
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    # Lowercase only ASCII/Latin characters; leave Devanagari casing alone
    s = "".join(ch.lower() if ch.isascii() else ch for ch in s)
    # Collapse whitespace
    return " ".join(s.split())


def compute_exact_hindi(gold: str, pred: str) -> int:
    """EM for Hindi using Hindi-aware normalization."""
    return int(normalize_hindi(gold) == normalize_hindi(pred))


def compute_f1_hindi(gold: str, pred: str) -> float:
    """Token-level F1 for Hindi using Hindi-aware normalization."""
    gold_toks = normalize_hindi(gold).split()
    pred_toks = normalize_hindi(pred).split()
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)



def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation / articles / extra whitespace (official SQuAD)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # both empty -> 1, otherwise 0
        return float(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_score_over_golds(pred: str, golds: list, metric_fn) -> float:
    """Max over all gold answers; unanswerable (no golds) is correct only if pred is empty."""
    if not golds:
        return metric_fn("", pred)
    return max(metric_fn(g, pred) for g in golds)


def aggregate(records):
    """
    records: list of dicts with keys has_gold (bool), em (float), f1 (float).
    Returns overall / HasAns / NoAns EM and F1 in percent.
    """
    def avg(rows, key):
        return 100.0 * sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    has = [r for r in records if r["has_gold"]]
    no = [r for r in records if not r["has_gold"]]
    return {
        "n": len(records),
        "n_has": len(has),
        "n_no": len(no),
        "EM": avg(records, "em"),
        "F1": avg(records, "f1"),
        "HasAns_EM": avg(has, "em"),
        "HasAns_F1": avg(has, "f1"),
        "NoAns_EM": avg(no, "em"),
        "NoAns_F1": avg(no, "f1"),
    }


def print_report(title, agg):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Examples: {agg['n']}  (answerable: {agg['n_has']}, unanswerable: {agg['n_no']})")
    print(f"Overall      EM: {agg['EM']:.1f}%   F1: {agg['F1']:.1f}%")
    print(f"Answerable   EM: {agg['HasAns_EM']:.1f}%   F1: {agg['HasAns_F1']:.1f}%")
    print(f"Unanswerable (correctly abstained): {agg['NoAns_EM']:.1f}%")
    print("=" * 70)
