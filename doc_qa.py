"""
doc_qa.py
---------
Document QA: retriever + reader pipeline (Part A).

WHY TWO STAGES?
The QAEngine.answer() takes a single passage.  For whole documents we first
retrieve the most relevant chunks (retriever.py), then run the reader on each
one.  Combining the results is not trivial:

  - If ANY chunk produces a non-abstained answer, prefer the highest-scoring one.
  - Abstain overall ONLY IF every chunk abstains (the question really isn't
    answerable from the whole document set).

This is the "best-wins" aggregation used in the original ORQA / DPR papers and
the HuggingFace pipeline.  We reuse decoding.decide() and the existing threshold
logic so the decision rule is identical to single-passage QA.
"""

from typing import List, Optional

from decoding import decide          # reuse the same decision function
from qa_engine import QAEngine
from retriever import Retriever, Chunk


class DocQA:
    """
    Two-stage Document QA.

    Usage
    -----
    dqa = DocQA()
    dqa.load(["report.pdf", "notes.txt"])
    result = dqa.ask("Who signed the contract?", top_k=5)
    """

    def __init__(self, qa_engine: Optional[QAEngine] = None,
                 retriever: Optional[Retriever] = None):
        # Allow injection for testing; otherwise build defaults lazily on first use.
        self._engine: Optional[QAEngine] = qa_engine
        self._retriever: Optional[Retriever] = retriever
        self._loaded_paths: List[str] = []

    # ── Lazy initialisation ──────────────────────────────────────────────────

    @property
    def engine(self) -> QAEngine:
        if self._engine is None:
            self._engine = QAEngine()
        return self._engine

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, paths: List[str], chunk_words: int = 200, overlap_words: int = 40):
        """Load and embed documents.  Call once; repeated calls with the same
        paths are free (memory cache in Retriever)."""
        self.retriever.load_files(paths, chunk_words=chunk_words,
                                  overlap_words=overlap_words)
        self._loaded_paths = paths

    def ask(self, question: str, top_k: int = 5,
            threshold: Optional[float] = None) -> dict:
        """
        Retrieve the top_k chunks and run the reader on each one.

        Returns
        -------
        dict with keys:
            answer          str   – best answer text, or "" if all abstained
            has_answer      bool
            source_chunk    Chunk – the chunk the answer came from (or best chunk)
            source_file     str   – path of that chunk's document
            retrieval_score float – cosine similarity of the source chunk
            confidence      float – reader confidence (from QAEngine)
            answerable_prob float – heuristic P(answerable)
            score_diff      float – null_score - best_span_score of winning chunk
            chunk_decisions list  – per-chunk detail dicts (for transparency)
        """
        if not self._loaded_paths:
            raise RuntimeError("Call load() with document paths before ask().")

        retrieved = self.retriever.retrieve(question, self._loaded_paths, top_k=top_k)
        if not retrieved:
            return self._empty_result()

        thr = threshold if threshold is not None else self.engine.threshold
        chunk_decisions = []

        for item in retrieved:
            chunk: Chunk = item["chunk"]
            r = self.engine.answer(question, chunk.text, threshold=thr)
            chunk_decisions.append({
                "chunk": chunk,
                "retrieval_score": item["score"],
                "rank": item["rank"],
                "has_answer": r["has_answer"],
                "answer": r["answer"],
                "confidence": r["confidence"],
                "answerable_prob": r["answerable_prob"],
                "score_diff": r["score_diff"],
                "candidate": r["candidate"],
                "start": r["start"],
                "end": r["end"],
            })

        # Aggregation: best-wins — pick the non-abstained answer with the
        # lowest (most negative) score_diff, i.e., the span that beats the
        # null score by the largest margin.
        answered = [d for d in chunk_decisions if d["has_answer"]]

        if answered:
            # sort by score_diff ascending (most negative = most confident span)
            best = min(answered, key=lambda d: d["score_diff"])
            return {
                "answer": best["answer"],
                "has_answer": True,
                "source_chunk": best["chunk"],
                "source_file": best["chunk"].source,
                "retrieval_score": best["retrieval_score"],
                "confidence": best["confidence"],
                "answerable_prob": best["answerable_prob"],
                "score_diff": best["score_diff"],
                "chunk_decisions": chunk_decisions,
            }
        else:
            # All chunks abstained — use the chunk with the smallest score_diff
            # (least confident abstain) as the "source" for transparency.
            best = min(chunk_decisions, key=lambda d: d["score_diff"])
            return {
                "answer": "",
                "has_answer": False,
                "source_chunk": best["chunk"],
                "source_file": best["chunk"].source,
                "retrieval_score": best["retrieval_score"],
                "confidence": best["confidence"],
                "answerable_prob": best["answerable_prob"],
                "score_diff": best["score_diff"],
                "chunk_decisions": chunk_decisions,
            }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        return {
            "answer": "", "has_answer": False,
            "source_chunk": None, "source_file": None,
            "retrieval_score": 0.0, "confidence": 1.0,
            "answerable_prob": 0.0, "score_diff": float("inf"),
            "chunk_decisions": [],
        }
