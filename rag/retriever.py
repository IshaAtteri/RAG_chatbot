from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    """Structured result for one retrieved chunk."""
    score: float
    text: str
    path: str      # filename only
    chunk_id: int  # index of chunk within that file


class BM25Retriever:
    """
    BM25-based retriever over a local text corpus.

    - Loads all .txt files from a corpus directory
    - Splits each file into overlapping word chunks
    - Builds a BM25 index over the chunks
    - retrieve(query, k) returns top-k chunks with scores + metadata
    """

    def __init__(
        self,
        corpus_dir: Path,
        chunk_size: int = 120,
        overlap: int = 30,
    ) -> None:
        """
        :param corpus_dir: Directory containing .txt files.
        :param chunk_size: Number of words per chunk.
        :param overlap: Number of overlapping words between consecutive chunks.
        """
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

        if not self.corpus_dir.exists():
            raise ValueError(f"Corpus directory does not exist: {self.corpus_dir}")

        # Internal storage
        self._chunks: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._tokenized_chunks: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        # Build index immediately on init
        self._build_index()

    # ------------------------
    # Public API
    # ------------------------

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks for a given query.

        :param query: User query string.
        :param k: Number of chunks to return.
        :return: List of RetrievedChunk objects sorted by score desc.
        """
        if not self._bm25 or not self._tokenized_chunks:
            # nothing indexed
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # sort by score descending, take top-k
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results: List[RetrievedChunk] = []
        for idx, score in ranked:
            meta = self._meta[idx]
            chunk_text = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    score=float(score),
                    text=chunk_text,
                    path=meta["path"],
                    chunk_id=meta["chunk_id"],
                )
            )

        return results

    # ------------------------
    # Internal helpers
    # ------------------------

    def _build_index(self) -> None:
        """Load all .txt files, chunk them, and build BM25 index."""
        chunks: List[str] = []
        meta: List[Dict[str, Any]] = []

        txt_files = list(self.corpus_dir.glob("*.txt"))
        if not txt_files:
            print(f"[BM25Retriever] Warning: no .txt files found in {self.corpus_dir}")

        for path in txt_files:
            text = path.read_text(encoding="utf-8")
            file_chunks = self._chunk_text(text)

            for i, ch in enumerate(file_chunks):
                chunks.append(ch)
                meta.append(
                    {
                        "path": path.name,
                        "chunk_id": i,
                    }
                )

        self._chunks = chunks
        self._meta = meta

        # Tokenize and build BM25
        self._tokenized_chunks = [c.lower().split() for c in self._chunks]
        if self._tokenized_chunks:
            self._bm25 = BM25Okapi(self._tokenized_chunks)
            print(
                f"[BM25Retriever] Indexed {len(self._chunks)} chunks "
                f"from {len(txt_files)} files in {self.corpus_dir}"
            )
        else:
            self._bm25 = None
            print("[BM25Retriever] Warning: no chunks were created; index is empty.")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word chunks."""
        words = text.split()
        chunks: List[str] = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            # slide window with overlap
            start += self.chunk_size - self.overlap

        return chunks
