import argparse
import os
import sys
from pathlib import Path

# Reduce llama.cpp noise
os.environ.setdefault("LLAMA_LOG_LEVEL", "NONE")

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from rag.rag_engine import RAGEngine  # type: ignore
from llama_cpp import Llama  # type: ignore


# ---------------- Guardrails ---------------- #

BANNED_PHRASES = [
    # self-harm
    "suicide",
    "kill myself",
    "harm myself",
    "self-harm",
    # explicit how-to violence / weapons (keep simple)
    "how to make a bomb",
    "build a bomb",
    "make explosives",
    # obvious crime
    "commit fraud",
    "steal credit card",
    "hack into",
]


def violates_guardrails(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in BANNED_PHRASES)


# ---------------- CLI + REPL ---------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local RAG chatbot (Phi-3 + BM25) – runs fully on your laptop."
    )

    # Model selection
    p.add_argument(
        "--model",
        choices=["q4", "fp16"],
        default="q4",
        help="Which Phi-3 model to use (default: q4).",
    )

    # RAG vs plain LLM
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable retrieval; use plain LLM only (no corpus).",
    )

    # Generation params
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per answer (default: 256).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9).",
    )

    # Retrieval params
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of retrieved chunks to use in RAG (default: 3).",
    )

    # Guardrails
    p.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Disable basic safety guardrails.",
    )

    # Paths (override defaults if needed)
    p.add_argument(
        "--models-dir",
        type=str,
        default=str(ROOT / "models"),
        help="Directory containing GGUF model files (default: ./models).",
    )
    p.add_argument(
        "--corpus-dir",
        type=str,
        default=str(ROOT / "corpus"),
        help="Directory containing text/pdf corpus files (default: ./corpus).",
    )

    return p


def select_model_path(args: argparse.Namespace) -> Path:
    models_dir = Path(args.models_dir)
    if args.model == "q4":
        candidate = models_dir / "Phi-3-mini-q4.gguf"
    else:
        candidate = models_dir / "Phi-3-mini-fp16.gguf"

    return candidate


def init_rag_engine(args: argparse.Namespace, model_path: Path) -> RAGEngine:
    corpus_dir = Path(args.corpus_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure the GGUF file exists in: {args.models_dir}"
        )

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Create it and add .txt/.pdf files, or adjust --corpus-dir."
        )

    engine = RAGEngine(
        model_path=model_path,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )
    return engine


def init_plain_llm(args: argparse.Namespace, model_path: Path) -> Llama:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure the GGUF file exists in: {args.models_dir}"
        )

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=8,
        verbose=False,
    )
    return llm


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = select_model_path(args)
    corpus_dir = Path(args.corpus_dir)

    print("=== Local RAG Chatbot ===")
    print(f"Model:       {model_path}")
    print(f"Corpus dir:  {corpus_dir}")
    print(f"Mode:        {'Plain LLM (no RAG)' if args.no_rag else 'RAG (BM25 + context)'}")
    print(f"Guardrails:  {'OFF' if args.no_guardrails else 'ON'}")
    print("Commands:    /exit, /quit, /help")
    print()

    # Initialize engines with error handling
    rag_engine = None
    llm_plain = None

    try:
        if not args.no_rag:
            rag_engine = init_rag_engine(args, model_path)
        else:
            llm_plain = init_plain_llm(args, model_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model or retrieval: {e}")
        sys.exit(1)

    # --------------- REPL loop --------------- #
    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        # Commands
        if user_q.lower() in {"/exit", "exit", "/quit", "quit"}:
            print("Goodbye!")
            break
        if user_q.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /exit or /quit  - leave the chat")
            print("  /help           - show this help")
            print("\nTips:")
            print("  Use --no-rag to compare plain LLM vs RAG.")
            print("  Use --model fp16 to run the FP16 baseline.\n")
            continue
        if not user_q:
            continue

        # Guardrails
        if not args.no_guardrails and violates_guardrails(user_q):
            print(
                "Assistant: I'm not able to help with that topic. "
                "Please ask about something else.\n"
            )
            continue

        # --------------- Generation --------------- #

        print("\n[Processing your request... please wait]\n")
        
        try:
            if args.no_rag:
                # Plain LLM mode (no retrieval)
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer clearly and concisely.",
                    },
                    {"role": "user", "content": user_q},
                ]
                out = llm_plain.create_chat_completion(
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                answer = out["choices"][0]["message"]["content"].strip()
                print("\nAssistant:")
                print(answer)
                print()
            else:
                # RAG mode
                answer, chunks = rag_engine.answer(
                    user_q,
                    k=args.k,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                print("\nAssistant:")
                print(answer)

                print("\nSources:")
                if chunks:
                    for i, ch in enumerate(chunks, start=1):
                        print(
                            f"  [{i}] {ch.path} "
                            f"(chunk {ch.chunk_id}, score={ch.score:.2f})"
                        )
                else:
                    print("  (no relevant sources found)")
                print()

        except Exception as e:
            print(f"\n[ERROR] Failed to generate answer: {e}\n")


if __name__ == "__main__":
    main()
