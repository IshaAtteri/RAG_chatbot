from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# Silence llama.cpp backend logs
os.environ["LLAMA_LOG_LEVEL"] = "NONE"

from llama_cpp import Llama  # noqa: E402

from .retriever import BM25Retriever, RetrievedChunk
from .prompt_builder import build_chat_messages


def _postprocess_answer(text: str) -> str:
    """
    Clean up some unwanted patterns the model sometimes adds
    (e.g., 'Written by [Your Name]', unnecessary 'Note:' blocks).
    """
    # Remove the exact annoying phrase if present
    text = text.replace("Written by [Your Name]", "")

    # Optionally strip trailing 'Note: ...' section, if present
    lower = text.lower()
    note_idx = lower.rfind("\nnote:")
    if note_idx != -1:
        text = text[:note_idx].rstrip()

    return text.strip()


class RAGEngine:
    """
    High-level RAG engine that wires together:
      - BM25Retriever over a local corpus
      - Phi-3 (or any llama.cpp model) for generation
    """

    def __init__(
        self,
        model_path: Path,
        corpus_dir: Path,
        n_ctx: int = 2048,
        n_threads: int = 8,
        chunk_size: int = 120,
        overlap: int = 30,
    ) -> None:
        self.model_path = Path(model_path)
        self.corpus_dir = Path(corpus_dir)

        # Init retriever
        self.retriever = BM25Retriever(
            corpus_dir=self.corpus_dir,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Init LLM (quiet)
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,   # <--- suppress Python-side llama logs
        )

    def answer(
        self,
        question: str,
        k: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Full RAG call:
        - retrieve top-k chunks
        - build chat messages
        - call the model
        - return (answer, chunks)
        """
        # 1) retrieval
        chunks = self.retriever.retrieve(question, k=k)

        # 2) build chat messages with context + question
        messages = build_chat_messages(question, chunks)

        # 3) call LLM
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        raw_answer = output["choices"][0]["message"]["content"].strip()
        clean_answer = _postprocess_answer(raw_answer)
        return clean_answer, chunks
