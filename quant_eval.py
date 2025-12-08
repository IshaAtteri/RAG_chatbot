import json
import os
import sys
import time
from pathlib import Path

# Silence llama.cpp logs
os.environ["LLAMA_LOG_LEVEL"] = "NONE"

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
# ------------------------------------

from rag.rag_engine import RAGEngine  # type: ignore


def load_test_set(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_answer(answer: str, expected_keywords):
    """
    Simple scoring:
      - count how many expected_keywords appear (case-insensitive)
      - recall = hits / total
      - pass = at least half of the keywords appear
    """
    answer_l = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_l)
    total = len(expected_keywords)
    recall = hits / total if total > 0 else 0.0
    passed = hits >= max(1, total // 2)  # at least half the keywords
    return passed, recall, hits


def main():
    model_path_q4 = ROOT / "models" / "Phi-3-mini-q4.gguf"      # quantized
    model_path_fp16 = ROOT / "models" / "Phi-3-mini-fp16.gguf"  # baseline
    corpus_dir = ROOT / "corpus"
    test_path = ROOT / "eval" / "phi_test_set.json"

    print("=== Quantization Evaluation (RAG only) ===")
    print("Q4   model:", model_path_q4)
    print("FP16 model:", model_path_fp16)
    print("Corpus:    ", corpus_dir)
    print("Test set:  ", test_path)
    print()

    tests = load_test_set(test_path)

    # --- Initialize RAG engines ---
    rag_q4 = RAGEngine(
        model_path=model_path_q4,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )

    rag_fp16 = RAGEngine(
        model_path=model_path_fp16,
        corpus_dir=corpus_dir,
        n_ctx=2048,
        n_threads=8,
    )

    results = []

    for test in tests:
        qid = test["id"]
        question = test["question"]
        expected = test["expected_keywords"]

        print(f"=== {qid} ===")
        print("Question:", question)

        # ---------- RAG + Q4 ----------
        t0 = time.time()
        ans_q4, chunks_q4 = rag_q4.answer(
            question,
            k=3,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        t_q4 = time.time() - t0

        pass_q4, rec_q4, hits_q4 = score_answer(ans_q4, expected)

        # ---------- RAG + FP16 ----------
        t0 = time.time()
        ans_fp16, chunks_fp16 = rag_fp16.answer(
            question,
            k=3,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        t_fp16 = time.time() - t0

        pass_fp16, rec_fp16, hits_fp16 = score_answer(ans_fp16, expected)

        results.append(
            {
                "id": qid,
                "question": question,
                "q4": {
                    "answer": ans_q4,
                    "time": t_q4,
                    "hits": hits_q4,
                    "recall": rec_q4,
                    "pass": pass_q4,
                },
                "fp16": {
                    "answer": ans_fp16,
                    "time": t_fp16,
                    "hits": hits_fp16,
                    "recall": rec_fp16,
                    "pass": pass_fp16,
                },
            }
        )

        # --- Per-question summary (short) ---
        print("\nQ4:")
        print("  Time:   {:.2f}s".format(t_q4))
        print("  Hits:   {} / {}".format(hits_q4, len(expected)))
        print("  Recall: {:.2f}".format(rec_q4))
        print("  Pass:   {}".format(pass_q4))

        print("\nFP16:")
        print("  Time:   {:.2f}s".format(t_fp16))
        print("  Hits:   {} / {}".format(hits_fp16, len(expected)))
        print("  Recall: {:.2f}".format(rec_fp16))
        print("  Pass:   {}".format(pass_fp16))
        print()

    # --- Aggregate summary ---
    n = len(results)

    def agg(key):
        passes = sum(1 for r in results if r[key]["pass"])
        avg_recall = sum(r[key]["recall"] for r in results) / n
        avg_time = sum(r[key]["time"] for r in results) / n
        return passes, avg_recall, avg_time

    q4_pass, q4_rec, q4_time = agg("q4")
    fp16_pass, fp16_rec, fp16_time = agg("fp16")

    print("=== OVERALL SUMMARY ===")
    print(f"Num questions: {n}\n")

    print(
        "Q4:   passes = {}/{}, avg recall = {:.2f}, avg time = {:.2f}s".format(
            q4_pass, n, q4_rec, q4_time
        )
    )
    print(
        "FP16: passes = {}/{}, avg recall = {:.2f}, avg time = {:.2f}s".format(
            fp16_pass, n, fp16_rec, fp16_time
        )
    )


if __name__ == "__main__":
    main()
