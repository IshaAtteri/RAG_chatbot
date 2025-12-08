import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from rag.retriever import BM25Retriever

def main():
    corpus_dir = Path("corpus")  # you currently have files directly under corpus/
    retriever = BM25Retriever(corpus_dir=corpus_dir, chunk_size=120, overlap=30)

    queries = [
        "what is retrieval augmented generation",
        "why do we quantize the language model",
        "architecture of the rag system",
    ]

    for q in queries:
        print("=" * 70)
        print("Query:", q)
        results = retriever.retrieve(q, k=2)

        for r in results:
            print(f"\nSource: {r.path}  [chunk {r.chunk_id}]  (score={r.score:.2f})")
            print("Snippet:")
            print(r.text[:250], "...")
        print()

if __name__ == "__main__":
    main()
