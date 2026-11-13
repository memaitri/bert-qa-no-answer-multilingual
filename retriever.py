"""
retriever.py
------------
Document loader + dense retriever for the Document QA pipeline (Part A).

WHY A RETRIEVER?
The base QAEngine.answer() takes a single passage.  For real documents (a PDF,
a folder of notes) there is no single "passage" — we need to find which chunk
of text is most likely to contain the answer before running the reader.

This module does two things:
  1. Load & chunk documents (.txt or .pdf) into ~200-word pieces with ~40-word
     overlap so no sentence is cut across a boundary.
  2. Encode every chunk with a small sentence-embedding model, then answer a
     query by cosine similarity and return the top-k chunks.

The heavy import (sentence_transformers) is kept inside the class constructor
so this file can be imported without it, and test_all.py --offline can still
test the chunking logic.

Override the embedding model at runtime:
    RETRIEVER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python demo_app.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

RETRIEVER_MODEL = os.environ.get("RETRIEVER_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_WORDS = 200     # target chunk size in whitespace-split tokens
OVERLAP_WORDS = 40    # overlap between consecutive chunks


@dataclass
class Chunk:
    """One piece of a document, ready for the reader."""
    text: str                 # the actual text
    source: str               # source file path or name
    char_start: int           # character offset in the original document text
    char_end: int             # character offset end
    chunk_idx: int            # 0-based index within this document


# ── Document loading ─────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse runs of whitespace / control chars to a single space."""
    text = re.sub(r"[\r\n\t\f\v]+", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return _clean(f.read())


def load_pdf(path: str) -> str:
    """Extract text from a PDF using pypdf (page by page, joined with spaces)."""
    try:
        from pypdf import PdfReader  # lazy import — not needed offline
    except ImportError:
        raise ImportError("pypdf is required for PDF loading: pip install pypdf")
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(_clean(t))
    return " ".join(pages)


def load_document(path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use .txt or .pdf.")


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str,
               chunk_words: int = CHUNK_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> List[Chunk]:
    """
    Split *text* into overlapping chunks of ~chunk_words whitespace-tokens.

    Approach:
      - Split on whitespace to get tokens, remembering each token's character
        offset in the original string.
      - Slide a window of chunk_words tokens forward by (chunk_words -
        overlap_words) each step so consecutive chunks share overlap_words tokens
        at their boundary.  This prevents answer spans from being cut in half.

    Returns a list of Chunk objects with correct char_start / char_end offsets.
    """
    if not text.strip():
        return []

    # Build list of (token, char_start) pairs using re.finditer for exact offsets
    tokens = [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]
    if not tokens:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks = []
    idx = 0

    while idx < len(tokens):
        window = tokens[idx: idx + chunk_words]
        first_token, cs = window[0]
        last_token, ls = window[-1]
        ce = ls + len(last_token)
        chunk_text_str = text[cs:ce]

        chunks.append(Chunk(
            text=chunk_text_str,
            source=source,
            char_start=cs,
            char_end=ce,
            chunk_idx=len(chunks),
        ))
        idx += step

    return chunks


# ── Retriever ────────────────────────────────────────────────────────────────

class Retriever:
    """
    Dense retriever: encode chunks with a sentence-transformer and retrieve
    top-k by cosine similarity.

    Embeddings are cached in memory — loading the same set of files twice
    returns the cached result without re-encoding.
    """

    def __init__(self, model_name: str = RETRIEVER_MODEL):
        # Heavy import kept local — this lets test_all.py --offline import
        # the chunking functions above without sentence_transformers installed.
        from sentence_transformers import SentenceTransformer
        print(f"Loading retriever model {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # cache: tuple(sorted file paths) -> (chunks, embeddings)
        self._cache: dict = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def load_files(self, paths: List[str],
                   chunk_words: int = CHUNK_WORDS,
                   overlap_words: int = OVERLAP_WORDS):
        """
        Load and chunk all files, embed every chunk, cache the result.
        Call this once per document set; subsequent calls with the same files
        are instant (cache hit).
        """
        key = tuple(sorted(os.path.abspath(p) for p in paths))
        if key in self._cache:
            chunks, _ = self._cache[key]
            print(f"  Cache hit: {len(chunks)} chunks from {len(paths)} file(s).")
            return

        all_chunks: List[Chunk] = []
        for p in paths:
            print(f"  Loading {os.path.basename(p)} ...")
            text = load_document(p)
            all_chunks.extend(chunk_text(text, source=p,
                                         chunk_words=chunk_words,
                                         overlap_words=overlap_words))

        print(f"  Embedding {len(all_chunks)} chunks ...")
        texts = [c.text for c in all_chunks]
        embeddings = self._embed(texts)          # (n_chunks, dim)
        self._cache[key] = (all_chunks, embeddings)
        print(f"  Done. {len(all_chunks)} chunks from {len(paths)} file(s).")

    def retrieve(self, query: str, paths: List[str],
                 top_k: int = 5) -> List[dict]:
        """
        Return the top_k most relevant chunks for *query* from the already-loaded
        *paths*.  Call load_files() first.

        Returns a list of dicts:
            chunk       Chunk object
            score       cosine similarity (0–1 for normalised embeddings)
            rank        1-based rank
        """
        key = tuple(sorted(os.path.abspath(p) for p in paths))
        if key not in self._cache:
            raise RuntimeError("Call load_files() before retrieve().")
        chunks, embeddings = self._cache[key]
        if not chunks:
            return []

        q_emb = self._embed([query])             # (1, dim)
        # cosine similarity: since bge models return unit-norm vectors this is
        # just the dot product.
        scores = (embeddings @ q_emb.T).squeeze()   # (n_chunks,)
        if scores.ndim == 0:                        # single-chunk edge case
            scores = scores.reshape(1)

        k = min(top_k, len(chunks))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            {"chunk": chunks[i], "score": float(scores[i]), "rank": rank + 1}
            for rank, i in enumerate(top_idx)
        ]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts and return a normalised float32 array (n, dim)."""
        embs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.array(embs, dtype=np.float32)
