from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from pypdf import PdfReader  # <--- NEW


@dataclass
class RetrievedChunk:
    path: str
    chunk_id: int
    text: str
    score: float


class BM25Retriever:
    """
    BM25-based retriever over a local corpus.

    Now supports:
      - .txt files
      - .pdf files (text-based PDFs)
    """

    def __init__(
        self,
        corpus_dir: Path,
        chunk_size: int = 120,
        overlap: int = 30,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.docs: List[Tuple[Path, str]] = []
        self.chunks: List[Tuple[Path, int, str]] = []  # (path, chunk_id, text)
        self.tokenized_chunks: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

        self._load_corpus()
        self._build_index()

    # ---------- File loading helpers ----------

    def _iter_corpus_files(self) -> List[Path]:
        """Return all .txt and .pdf files under corpus_dir (recursively)."""
        exts = {".txt", ".pdf"}
        files = [
            p for p in self.corpus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
        return sorted(files)

    def _read_text_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf_file(self, path: Path) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            print(f"[BM25Retriever] WARNING: failed to open PDF {path.name}: {e}")
            return ""

        pages_text: List[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages_text.append(text)

        return "\n\n".join(pages_text)

    def _read_doc(self, path: Path) -> str:
        """Dispatch reader based on file extension."""
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._read_text_file(path)
        elif suffix == ".pdf":
            return self._read_pdf_file(path)
        else:
            return ""

    def _load_corpus(self) -> None:
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        files = self._iter_corpus_files()
        if not files:
            print(f"[BM25Retriever] No .txt or .pdf files found in {self.corpus_dir}")
            return

        for path in files:
            text = self._read_doc(path)
            if not text.strip():
                # skip empty or unreadable docs
                print(f"[BM25Retriever] Skipping empty or unreadable file: {path.name}")
                continue
            self.docs.append((path, text))

    # ---------- Chunking & indexing ----------

    def _chunk_text(self, text: str) -> List[str]:
        """
        Simple word-based chunking with overlap.
        """
        words = text.split()
        chunks: List[str] = []

        if not words:
            return chunks

        step = self.chunk_size - self.overlap
        for start in range(0, len(words), step):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words))

        return chunks

    def _build_index(self) -> None:
        """
        Build BM25 index over all chunks from all docs.
        """
        all_chunks: List[str] = []
        chunk_meta: List[Tuple[Path, int, str]] = []

        for path, text in self.docs:
            doc_chunks = self._chunk_text(text)
            for i, ch in enumerate(doc_chunks):
                chunk_meta.append((path, i, ch))
                all_chunks.append(ch)

        if not all_chunks:
            print("[BM25Retriever] WARNING: no chunks were created from corpus.")
            return

        # Tokenize: lowercase whitespace split
        tokenized = [ch.lower().split() for ch in all_chunks]

        self.chunks = chunk_meta
        self.tokenized_chunks = tokenized
        self.bm25 = BM25Okapi(self.tokenized_chunks)

        print(
            f"[BM25Retriever] Indexed {len(self.chunks)} chunks "
            f"from {len(self.docs)} files in corpus"
        )

    # ---------- Retrieval ----------

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks for a query.
        """
        if self.bm25 is None or not self.chunks:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # pair scores with indices
        scored_idx = list(enumerate(scores))
        scored_idx.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievedChunk] = []
        for idx, score in scored_idx[:k]:
            path, chunk_id, text = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    path=str(path.name),
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                )
            )
        return results
