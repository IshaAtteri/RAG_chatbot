import json
import sys
import time
from pathlib import Path

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
# ------------------------------------

from rag.rag_engine import RAGEngine  # type: ignore
from llama_cpp import Llama  # type: ignore


def load_test_set(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_answer(answer: str, expected_keywords):
    """
    Very simple scoring:
      - count how many expected_keywords appear (case-insensitive)
      - compute recall as hits / total
      - consider 'pass' if at least 50% of keywords appear
    """
    answer_l = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_l)
    total = len(expected_keywords)
    recall = hits / total if total > 0 else 0.0
    passed = hits >= max(1, total // 2)  # at least half the keywords
    return passed, recall, hits


def main():
    model_path_q4 = ROOT / "models" / "Phi-3-mini-q4.gguf"      # quantized
    corpus_dir = ROOT / "corpus"
    test_path = ROOT / "eval" / "test_set.json"

    print("Model (Q4):", model_path_q4)
    print("Corpus:", corpus_dir)
    print("Test set:", test_path)
    print()

    tests = load_test_set(test_path)

    # --- Initialize engines ---
    # RAG engine (Q4)
    rag_engine = RAGEngine(
        model_path=model_path_q4,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )

    # Plain LLM (no RAG) using same quantized model
    llm_plain = Llama(
        model_path=str(model_path_q4),
        n_ctx=2048,
        n_threads=8,
        verbose=False,
    )

    results = []

    for test in tests:
        qid = test["id"]
        question = test["question"]
        expected = test["expected_keywords"]

        print(f"=== {qid} ===")
        print("Question:", question)

        # --- Plain LLM baseline (no RAG) ---
        t0 = time.time()
        messages_plain = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question clearly and concisely."
                ),
            },
            {"role": "user", "content": question},
        ]
        out_plain = llm_plain.create_chat_completion(
            messages=messages_plain,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        ans_plain = out_plain["choices"][0]["message"]["content"].strip()
        t_plain = time.time() - t0

        pass_plain, recall_plain, hits_plain = score_answer(ans_plain, expected)

        # --- RAG-enabled answer ---
        t0 = time.time()
        ans_rag, chunks = rag_engine.answer(
            question,
            k=3,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        t_rag = time.time() - t0

        pass_rag, recall_rag, hits_rag = score_answer(ans_rag, expected)

        results.append(
            {
                "id": qid,
                "question": question,
                "plain": {
                    "answer": ans_plain,
                    "time": t_plain,
                    "hits": hits_plain,
                    "recall": recall_plain,
                    "pass": pass_plain,
                },
                "rag": {
                    "answer": ans_rag,
                    "time": t_rag,
                    "hits": hits_rag,
                    "recall": recall_rag,
                    "pass": pass_rag,
                },
            }
        )

        # --- Print short per-question summary ---
        print("\nPlain LLM:")
        print("  Time:   {:.2f}s".format(t_plain))
        print("  Hits:   {} / {}".format(hits_plain, len(expected)))
        print("  Recall: {:.2f}".format(recall_plain))
        print("  Pass:   {}".format(pass_plain))

        print("\nRAG (BM25 + context):")
        print("  Time:   {:.2f}s".format(t_rag))
        print("  Hits:   {} / {}".format(hits_rag, len(expected)))
        print("  Recall: {:.2f}".format(recall_rag))
        print("  Pass:   {}".format(pass_rag))
        print()

    # --- Aggregate summary ---
    n = len(results)
    plain_pass = sum(1 for r in results if r["plain"]["pass"])
    rag_pass = sum(1 for r in results if r["rag"]["pass"])

    avg_plain_recall = sum(r["plain"]["recall"] for r in results) / n
    avg_rag_recall = sum(r["rag"]["recall"] for r in results) / n

    avg_plain_time = sum(r["plain"]["time"] for r in results) / n
    avg_rag_time = sum(r["rag"]["time"] for r in results) / n

    print("=== OVERALL SUMMARY ===")
    print(f"Num questions: {n}")
    print(f"Plain LLM: passes = {plain_pass}/{n}, avg recall = {avg_plain_recall:.2f}, avg time = {avg_plain_time:.2f}s")
    print(f"RAG:       passes = {rag_pass}/{n}, avg recall = {avg_rag_recall:.2f}, avg time = {avg_rag_time:.2f}s")


if __name__ == "__main__":
    main()
