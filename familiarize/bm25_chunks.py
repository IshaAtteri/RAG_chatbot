from pathlib import Path
from rank_bm25 import BM25Okapi

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
CORPUS_DIR = ROOT_DIR / "corpus"   # you are using plain 'corpus' now

def chunk_text(text, chunk_size=120, overlap=30):
    """
    Split text into overlapping word chunks:
    - chunk_size: number of words per chunk
    - overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap  # slide window with overlap

    return chunks

def load_corpus_chunks():
    all_chunks = []
    meta = []

    for path in CORPUS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=120, overlap=30)

        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "path": path.name,
                "chunk_id": i,
            })

    return all_chunks, meta

def build_bm25(chunks):
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def retrieve_chunks(bm25, tokenized_chunks, meta, query, k=3):
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in ranked:
        results.append({
            "score": float(score),
            "meta": meta[idx],
            "text": " ".join(tokenized_chunks[idx]),
        })
    return results

def main():
    print("Corpus dir:", CORPUS_DIR)

    chunks, meta = load_corpus_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(set(m['path'] for m in meta))} files.\n")

    bm25, tokenized_chunks = build_bm25(chunks)

    queries = [
        "what is retrieval augmented generation",
        "why do we quantize the language model",
        "architecture of the rag system",
    ]

    for q in queries:
        print("=" * 70)
        print("Query:", q)
        results = retrieve_chunks(bm25, tokenized_chunks, meta, q, k=2)

        for r in results:
            print(f"\nSource: {r['meta']['path']}  [chunk {r['meta']['chunk_id']}]  (score={r['score']:.2f})")
            print("Snippet:")
            print(r["text"][:250], "...")
        print()

if __name__ == "__main__":
    main()
