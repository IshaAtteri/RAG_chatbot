import sys
from pathlib import Path

# --- make project root importable ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# ------------------------------------

from rag.rag_engine import RAGEngine  # type: ignore


def main():
    model_path = ROOT / "models" / "Phi-3-mini-q4.gguf"    # adjust if filename differs
    corpus_dir = ROOT / "corpus"

    print("Model:", model_path)
    print("Corpus:", corpus_dir)

    engine = RAGEngine(
        model_path=model_path,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )

    question = "What is retrieval augmented generation and how does this project implement it?"
    print("\nQUESTION:")
    print(question)

    answer, chunks = engine.answer(question, k=3, max_tokens=256)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== SOURCES ===")
    for i, ch in enumerate(chunks, start=1):
        print(f"[Source {i}] {ch.path} (chunk {ch.chunk_id}, score={ch.score:.2f})")

if __name__ == "__main__":
    main()
