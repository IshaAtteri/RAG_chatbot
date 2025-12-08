import sys
from pathlib import Path

# --- make project root importable when run as "python rag_cli.py" ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
# --------------------------------------------------------------------

from rag.rag_engine import RAGEngine  # type: ignore


def main():
    model_path = ROOT / "models" / "Phi-3-mini-q4.gguf"  # adjust if needed
    corpus_dir = ROOT / "corpus"

    print("=== Local RAG CLI ===")
    print(f"Model:  {model_path}")
    print(f"Corpus: {corpus_dir}")
    print("Type 'exit' or 'quit' to stop.\n")

    engine = RAGEngine(
        model_path=model_path,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )

    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_q:
            continue

        # Simple guardrail example: refuse obviously out-of-scope stuff
        banned = ["self-harm", "suicide", "kill myself"]
        if any(b in user_q.lower() for b in banned):
            print("Assistant: I'm not able to help with that topic.\n")
            continue

        answer, chunks = engine.answer(user_q, k=3, max_tokens=256)

        print("\nAssistant:")
        print(answer)

        print("\nSources:")
        for i, ch in enumerate(chunks, start=1):
            print(f"  [{i}] {ch.path} (chunk {ch.chunk_id}, score={ch.score:.2f})")
        print()


if __name__ == "__main__":
    main()
