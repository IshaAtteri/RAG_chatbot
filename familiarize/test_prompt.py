import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from rag.retriever import BM25Retriever
from rag.prompt_builder import build_rag_prompt, build_chat_messages


def main():
    corpus_dir = Path("corpus")  # adjust if you move corpus
    retriever = BM25Retriever(corpus_dir=corpus_dir, chunk_size=120, overlap=30)

    query = "What is retrieval augmented generation and how does this project implement it?"

    # 1) retrieve chunks
    chunks = retriever.retrieve(query, k=3)

    # 2) build plain prompt
    prompt = build_rag_prompt(query, chunks)
    print("===== RAG PROMPT (plain) =====")
    print(prompt)
    print()

    # 3) build chat messages
    messages = build_chat_messages(query, chunks)
    print("===== CHAT MESSAGES =====")
    for m in messages:
        print(f"[{m['role'].upper()}]\n{m['content']}\n")

if __name__ == "__main__":
    main()
