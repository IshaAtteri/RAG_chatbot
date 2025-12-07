from pathlib import Path
from rank_bm25 import BM25Okapi

# Paths
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
CORPUS_DIR = ROOT_DIR / "corpus"

def load_corpus():
    docs = []
    meta = []
    for path in CORPUS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(text)
        meta.append({"path": path.name})
    return docs, meta

def build_bm25(docs):
    # super simple tokenization: lowercase + split on whitespace
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def retrieve(bm25, tokenized_docs, meta, query, k=3):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    # sort by score descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in ranked:
        results.append({
            "score": float(score),
            "meta": meta[idx],
            "text": tokenized_docs[idx],  # tokens for now
        })
    return results

def main():
    print("Corpus dir:", CORPUS_DIR)

    docs, meta = load_corpus()
    print(f"Loaded {len(docs)} documents.\n")

    bm25, tokenized_docs = build_bm25(docs)

    # Try a few queries
    queries = [
        "what is retrieval augmented generation",
        "explain language models and transformers",
        "gait events and CNN BiLSTM",
    ]

    for q in queries:
        print("=" * 60)
        print("Query:", q)
        results = retrieve(bm25, tokenized_docs, meta, q, k=2)

        for r in results:
            print(f"- {r['meta']['path']} (score={r['score']:.2f})")
        print()

if __name__ == "__main__":
    main()
