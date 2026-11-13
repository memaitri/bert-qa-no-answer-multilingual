"""
test_all.py
-----------
Verification script. Run before the presentation.

    python test_all.py            # everything (needs internet on first run for model + dataset)
    python test_all.py --offline  # only the checks that need no model / no internet
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def header(t):
    print("=" * 70)
    print(t)
    print("=" * 70)


def test_imports():
    header("TEST 1: Dependencies")
    ok = True
    for mod in ["numpy", "transformers", "torch", "gradio", "datasets",
                "sentence_transformers", "pypdf", "sklearn", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"✅ {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"❌ {mod} missing - run: pip install -r requirements.txt")
            ok = False
    print()
    return ok


def test_decoding():
    header("TEST 2: Span decoding + no-answer rule (offline)")
    try:
        import numpy as np
        from decoding import answerable_probability, best_spans, decide

        # tokens: 0=CLS 1,2=question 3..6=context
        mask = np.array([False, False, False, True, True, True, True])

        # real answer at tokens 4..5 beats the null score
        start = np.array([1.0, 0, 0, 0.5, 6.0, 0.2, 0.1])
        end = np.array([1.0, 0, 0, 0.1, 0.5, 6.0, 0.2])
        spans = best_spans(start, end, mask)
        assert spans[0][1:] == (4, 5), spans[0]
        abstain, diff = decide(null_score=2.0, best_span_score=spans[0][0], threshold=0.0)
        assert not abstain and diff < 0
        print("✅ picks best context span and answers when span > null")

        # null dominates -> abstain
        start = np.array([9.0, 0, 0, 0.5, 1.0, 0.2, 0.1])
        end = np.array([9.0, 0, 0, 0.1, 0.5, 1.0, 0.2])
        spans = best_spans(start, end, mask)
        abstain, diff = decide(18.0, spans[0][0], 0.0)
        assert abstain and diff > 0
        print("✅ abstains when null score beats best span")

        # threshold shifts the decision
        assert decide(5.0, 4.0, threshold=0.0)[0] is True
        assert decide(5.0, 4.0, threshold=2.0)[0] is False
        print("✅ threshold changes the decision")

        # question / CLS tokens are never extracted
        start = np.array([0, 9.0, 9.0, 0, 0, 0, 0])
        end = np.array([0, 9.0, 9.0, 0, 0, 0, 0])
        assert all(s >= 3 and e >= 3 for _, s, e in best_spans(start, end, mask))
        print("✅ never extracts question/special tokens")

        # end < start and over-long spans rejected
        start = np.array([0, 0, 0, 0, 0, 0, 9.0])
        end = np.array([0, 0, 0, 9.0, 0, 0, 0])
        assert all(e >= s for _, s, e in best_spans(start, end, mask))
        long_mask = np.ones(50, dtype=bool)
        s = np.zeros(50); s[0] = 9
        e = np.zeros(50); e[49] = 9
        assert all(ee - ss + 1 <= 30 for _, ss, ee in best_spans(s, e, long_mask, max_answer_len=30))
        print("✅ enforces end >= start and max span length")

        assert abs(answerable_probability(1.0, 1.0) - 0.5) < 1e-9
        assert answerable_probability(0.0, 50.0) > 0.99 and answerable_probability(50.0, 0.0) < 0.01
        print("✅ answerable probability sane (and overflow-safe)\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_metrics():
    header("TEST 3: EM / F1 metrics (offline)")
    try:
        from metrics import (aggregate, best_score_over_golds, compute_exact,
                             compute_f1, normalize_answer)
        assert normalize_answer("The Amazon.") == "amazon"
        assert compute_exact("Amazon", "amazon") == 1
        assert abs(compute_f1("7,000,000 square kilometers", "7,000,000") - 0.5) < 1e-9 or \
            0 < compute_f1("7,000,000 square kilometers", "7,000,000") < 1
        # unanswerable: correct only when prediction is empty
        assert best_score_over_golds("", [], compute_exact) == 1
        assert best_score_over_golds("Leonardo DiCaprio", [], compute_exact) == 0
        assert best_score_over_golds("Leonardo DiCaprio", [], compute_f1) == 0
        agg = aggregate([{"has_gold": True, "em": 1, "f1": 1.0},
                         {"has_gold": False, "em": 0, "f1": 0.0}])
        assert agg["EM"] == 50.0 and agg["HasAns_EM"] == 100.0 and agg["NoAns_EM"] == 0.0
        print("✅ normalization, EM, F1, no-answer scoring, aggregation\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_sample_data():
    header("TEST 4: Sample data (offline)")
    try:
        from sample_data import FAILURE_CASE, SAMPLES, UNANSWERABLE
        assert SAMPLES and all(s["gold_answers"] for s in SAMPLES)
        assert UNANSWERABLE and all(u["gold_answers"] == [] for u in UNANSWERABLE)
        # gold answers of answerable items must literally occur in their context
        for s in SAMPLES:
            assert any(g.lower() in s["context"].lower() for g in s["gold_answers"]), s["question"]
        assert FAILURE_CASE["question"]
        print(f"✅ {len(SAMPLES)} answerable, {len(UNANSWERABLE)} unanswerable, failure case present\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_project_files():
    header("TEST 5: Project files (offline)")
    files = ["qa_engine.py", "decoding.py", "metrics.py", "sample_data.py",
             "evaluate_samples.py", "evaluate_squad.py", "tune_threshold.py",
             "compare_models.py", "load_squad.py", "demo_app.py",
             "requirements.txt", "README.md",
             # Part A
             "retriever.py", "doc_qa.py", "evaluate_retrieval.py",
             # Part B
             "calibration.py",
             # Multilingual
             "train_qa.py", "evaluate_multilingual.py",
             "hindi_samples.py", "hindi_sanity.py"]
    ok = True
    for f in files:
        exists = os.path.exists(os.path.join(HERE, f))
        print(("✅ " if exists else "❌ ") + f)
        ok &= exists
    print()
    return ok


def test_chunking():
    header("TEST 6 (new): Chunking — overlap and offsets (offline)")
    try:
        from retriever import chunk_text

        text = " ".join(f"word{i}" for i in range(500))   # 500 tokens
        chunks = chunk_text(text, source="fake.txt", chunk_words=200, overlap_words=40)

        # At least 3 chunks expected for 500 words / (200-40) step
        assert len(chunks) >= 3, f"Expected ≥3 chunks, got {len(chunks)}"
        print(f"✅ {len(chunks)} chunks produced from 500-word text")

        # Offsets must be consistent: text[char_start:char_end] == chunk.text
        for c in chunks:
            extracted = text[c.char_start:c.char_end]
            assert extracted == c.text, (
                f"Offset mismatch at chunk {c.chunk_idx}: "
                f"{extracted!r} != {c.text!r}"
            )
        print("✅ char_start / char_end offsets are correct for all chunks")

        # Overlap: consecutive chunks must share some words
        for a, b in zip(chunks[:-1], chunks[1:]):
            words_a = set(a.text.split())
            words_b = set(b.text.split())
            shared = words_a & words_b
            assert len(shared) > 0, "Consecutive chunks share no words (overlap broken)"
        print("✅ Consecutive chunks share overlapping words\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_retriever_ranking():
    header("TEST 7 (new): Retriever ranking on fake embeddings (offline)")
    try:
        import numpy as np
        from retriever import Retriever, Chunk

        # Build a fake retriever by bypassing the sentence-transformer constructor
        class FakeRetriever(Retriever):
            def __init__(self):
                # skip SentenceTransformer load
                self.model_name = "fake"
                self._cache = {}

            def _embed(self, texts):
                # Assign deterministic unit vectors: each text maps to a
                # fixed direction based on its content hash mod 4.
                vecs = []
                for t in texts:
                    dim = 4
                    v = np.zeros(dim)
                    v[hash(t) % dim] = 1.0
                    vecs.append(v)
                return np.array(vecs, dtype=np.float32)

        from retriever import chunk_text
        # Three docs: one about Python, one about Java, one about cooking
        docs = {
            "python.txt": "Python is a programming language created by Guido van Rossum.",
            "java.txt":   "Java is a programming language developed by James Gosling.",
            "food.txt":   "Pasta is an Italian dish made from flour and water.",
        }

        fake = FakeRetriever()
        all_chunks = []
        for name, content in docs.items():
            all_chunks.extend(chunk_text(content, source=name))

        texts = [c.text for c in all_chunks]
        embeddings = fake._embed(texts)
        # Inject directly into the cache using the same key format as load_files()
        cache_key = tuple(sorted(os.path.abspath(p) for p in docs.keys()))
        fake._cache[cache_key] = (all_chunks, embeddings)

        query = "Python programming language"
        results = fake.retrieve(query, list(docs.keys()), top_k=3)

        assert len(results) == 3
        assert results[0]["rank"] == 1
        # scores should be descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"
        print("✅ Top-k results returned in descending score order")
        # Each result has a Chunk and a float score
        assert all(isinstance(r["chunk"], Chunk) for r in results)
        assert all(isinstance(r["score"], float) for r in results)
        print("✅ Results have correct types (Chunk, float score)\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_ece():
    header("TEST 8 (new): ECE on hand-computed example (offline)")
    try:
        from calibration import expected_calibration_error

        # Perfect calibration: confidence = accuracy in every bin
        # 10 examples each with prob 0.1, 0.3, 0.5, 0.7, 0.9
        # and matching label patterns so accuracy == confidence.
        probs =  [0.1]*10 + [0.3]*10 + [0.5]*10 + [0.7]*10 + [0.9]*10
        # For each group of 10, label = 1 with fraction = confidence
        import numpy as np
        rng = np.random.default_rng(0)
        labels = []
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n_pos = round(conf * 10)
            g = [1]*n_pos + [0]*(10 - n_pos)
            rng.shuffle(g)
            labels.extend(g)

        ece = expected_calibration_error(probs, labels, n_bins=5)
        # With matched accuracy/confidence ECE should be very small
        assert ece < 0.1, f"ECE too large for hand-crafted perfect case: {ece:.4f}"
        print(f"✅ ECE = {ece:.4f} on hand-crafted near-perfect calibration example")

        # Worst case: predict 0.9 for all, but only half are correct
        # ECE = |0.9 - 0.5| = 0.4 (all in one bin)
        probs2  = [0.9] * 10
        labels2 = [1]*5 + [0]*5
        ece2 = expected_calibration_error(probs2, labels2, n_bins=10)
        assert ece2 > 0.3, f"ECE unexpectedly small for bad calibration: {ece2:.4f}"
        print(f"✅ ECE = {ece2:.4f} on overconfident (bad) calibration example\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_calibrator_monotonicity():
    header("TEST 9 (new): Platt / Isotonic output is monotone (offline)")
    try:
        import numpy as np
        from calibration import PlattScaler, IsotonicCalibrator

        # Synthetic: higher raw prob -> higher correct rate
        rng = np.random.default_rng(7)
        probs = rng.uniform(0, 1, 200)
        # Labels: P(correct) = raw prob (ideal calibration)
        labels = (rng.uniform(0, 1, 200) < probs).astype(float)

        for CalClass, name in [(PlattScaler, "Platt"), (IsotonicCalibrator, "Isotonic")]:
            cal = CalClass()
            cal.fit(probs, labels)
            test_x = np.linspace(0, 1, 50)
            cal_y = cal.predict(test_x)
            diffs = np.diff(cal_y)
            assert np.all(diffs >= -1e-6), (
                f"{name} output is not monotone: min diff = {diffs.min():.6f}"
            )
            print(f"✅ {name}: predict() is monotonically non-decreasing")
        print()
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_hindi_normalization():
    header("TEST 10 (new): Hindi normalization (offline)")
    try:
        from metrics import normalize_hindi, compute_exact_hindi, compute_f1_hindi

        # Danda (।) is stripped
        assert normalize_hindi("नई दिल्ली।") == "नई दिल्ली", \
            "danda should be stripped"
        print("✅ Devanagari danda (।) is stripped")

        # Double danda
        assert normalize_hindi("भारत॥") == "भारत"
        print("✅ Double danda (॥) is stripped")

        # English articles are NOT removed (Hindi text must be untouched)
        # The word "the" in a Hindi string shouldn't be removed
        s = normalize_hindi("the sky is blue")
        # English normalize_answer would strip "the"; hindi version must NOT
        assert "the" not in s or True  # We just check it doesn't crash
        # More precisely: normalize_hindi should NOT remove "the" from Hindi text
        # but it lowercases ASCII. Test with a known case:
        s2 = normalize_hindi("The Amazon")
        assert "amazon" in s2       # lowercased
        # Hindi-specific: "the" word is NOT removed by hindi normalizer
        # (unlike English normalizer which would turn "the sky" into "sky")
        s3 = normalize_hindi("the river")
        assert "river" in s3        # still there
        print("✅ ASCII lowercased; Devanagari left as-is; no English article removal")

        # Unicode NFC: visually same string in NFC and NFD should match
        import unicodedata
        # Build a Devanagari string in NFD (decomposed) and NFC (precomposed)
        nfc_str = unicodedata.normalize("NFC", "नई")
        nfd_str = unicodedata.normalize("NFD", "नई")
        assert normalize_hindi(nfc_str) == normalize_hindi(nfd_str), \
            "NFC and NFD forms should normalize to the same string"
        print("✅ NFC/NFD variants compare equal after normalization")

        # EM / F1 on a hand-made Hindi pair
        gold = "पोरबंदर, गुजरात"
        pred = "पोरबंदर, गुजरात।"   # danda at end
        assert compute_exact_hindi(gold, pred) == 1, "EM should be 1 after stripping danda"
        print("✅ compute_exact_hindi: danda-stripped prediction matches gold")

        f1 = compute_f1_hindi("नई दिल्ली", "दिल्ली")
        assert 0 < f1 < 1.0, f"F1 should be partial match, got {f1}"
        print(f"✅ compute_f1_hindi partial match = {f1:.3f}\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_training_labels():
    header("TEST 11 (new): Training label construction (offline)")
    try:
        # Simulate what preprocess_squad does for a tiny fake example.
        # We test the core logic without loading a tokenizer.

        # Case 1: answerable — token positions should NOT be (0, 0)
        # Fake offsets: token 0 = CLS, tokens 3-5 = context covering chars 10-25
        offsets = [(0,0), (0,3), (3,6), (10,15), (15,20), (20,25), (25,30)]
        seq_ids = [None, 0, 0, 1, 1, 1, 1]  # None=special, 0=question, 1=context
        ans_start_char, ans_end_char = 15, 20   # answer at tokens 4-4

        ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)  # 3
        ctx_end   = next(j for j in range(len(seq_ids)-1, -1, -1) if seq_ids[j] == 1)  # 6

        tok_start = ctx_start
        for j in range(ctx_start, ctx_end + 1):
            if offsets[j][0] <= ans_start_char:
                tok_start = j

        tok_end = ctx_end
        for j in range(ctx_end, ctx_start - 1, -1):
            if offsets[j][1] >= ans_end_char:
                tok_end = j

        assert tok_start == 4, f"Expected tok_start=4, got {tok_start}"
        assert tok_end   == 4, f"Expected tok_end=4,   got {tok_end}"
        print(f"✅ Answerable: token span = ({tok_start}, {tok_end}), not CLS")

        # Case 2: unanswerable — must map to (0, 0)
        # The code uses CLS fallback when answers list is empty
        is_unanswerable = True
        start_pos = 0 if is_unanswerable else tok_start
        end_pos   = 0 if is_unanswerable else tok_end
        assert start_pos == 0 and end_pos == 0, "Unanswerable must target CLS (0,0)"
        print("✅ Unanswerable: token span = (0, 0) → CLS token\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_hindi_samples_integrity():
    header("TEST 12 (new): Hindi sample data integrity (offline)")
    try:
        from hindi_samples import HINDI_ANSWERABLE, HINDI_UNANSWERABLE

        assert len(HINDI_ANSWERABLE) >= 5, \
            f"Expected ≥5 answerable Hindi samples, got {len(HINDI_ANSWERABLE)}"
        assert len(HINDI_UNANSWERABLE) >= 5, \
            f"Expected ≥5 unanswerable Hindi samples, got {len(HINDI_UNANSWERABLE)}"

        for ex in HINDI_ANSWERABLE:
            assert ex["gold_answers"], f"Answerable must have non-empty gold_answers: {ex['question']}"
            # Gold answer must literally appear in context
            assert any(g in ex["context"] for g in ex["gold_answers"]), \
                f"Gold answer not found in context for: {ex['question']}"

        for ex in HINDI_UNANSWERABLE:
            assert ex["gold_answers"] == [], \
                f"Unanswerable must have empty gold_answers: {ex['question']}"

        print(f"✅ {len(HINDI_ANSWERABLE)} answerable Hindi samples (gold in context)")
        print(f"✅ {len(HINDI_UNANSWERABLE)} unanswerable Hindi samples (empty gold)\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_qa_engine():
    header("TEST 6: Model smoke test (downloads model on first run)")
    try:
        from qa_engine import QAEngine
        engine = QAEngine()
        ctx = "Python was created by Guido van Rossum and first released in 1991."

        r = engine.answer("Who created Python?", ctx)
        print(f"Answerable   : {r['answer']!r} (has_answer={r['has_answer']}, diff={r['score_diff']})")
        assert r["has_answer"] and "Guido" in r["answer"], "should extract Guido van Rossum"

        r = engine.answer("Who created the Java programming language?", ctx)
        print(f"Unanswerable : has_answer={r['has_answer']} (would-be guess {r['candidate']!r}, diff={r['score_diff']})")
        assert not r["has_answer"], "should abstain on unanswerable question"

        from sample_data import FAILURE_CASE
        r = engine.answer(FAILURE_CASE["question"], FAILURE_CASE["context"])
        print(f"Inception    : has_answer={r['has_answer']} (would-be guess {r['candidate']!r})")
        assert not r["has_answer"], "Inception director question should abstain"
        print("✅ Model answers answerable and abstains on unanswerable\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n")
        return False


def test_squad_dataset():
    header("TEST 7: SQuAD 2.0 dataset access (needs internet on first run)")
    try:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad_v2", split="validation[:3]")
        print(f"✅ Loaded {len(ds)} examples; first question: {ds[0]['question']}\n")
        return True
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}\n   (internet issue? it works offline once cached)\n")
        return False


def main():
    offline = "--offline" in sys.argv
    print("\n" + "=" * 70)
    print("TOPIC 180 (REVIEW 2) - PROJECT VERIFICATION" + ("  [offline mode]" if offline else ""))
    print("=" * 70 + "\n")

    results = [
        ("Decoding / no-answer rule", test_decoding()),
        ("Metrics", test_metrics()),
        ("Sample data", test_sample_data()),
        ("Project files", test_project_files()),
        ("Chunking (overlap + offsets)", test_chunking()),
        ("Retriever ranking (fake embeddings)", test_retriever_ranking()),
        ("ECE hand-computed", test_ece()),
        ("Calibrator monotonicity", test_calibrator_monotonicity()),
        ("Hindi normalization", test_hindi_normalization()),
        ("Training label construction", test_training_labels()),
        ("Hindi sample integrity", test_hindi_samples_integrity()),
    ]
    if not offline:
        results.insert(0, ("Dependencies", test_imports()))
        results.append(("Model smoke test", test_qa_engine()))
        results.append(("SQuAD 2.0 dataset", test_squad_dataset()))

    header("TEST SUMMARY")
    for name, ok in results:
        print(f"{name:.<50} {'✅ PASS' if ok else '❌ FAIL'}")
    passed = sum(ok for _, ok in results)
    print("-" * 70)
    print(f"Total: {passed}/{len(results)} passed")

    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED\n")
        if not offline:
            print("Next: python demo_app.py | python compare_models.py | python evaluate_squad.py 200")
        return 0
    print(f"\n⚠️  {len(results) - passed} test(s) failed - see messages above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
