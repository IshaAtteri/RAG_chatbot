Local RAG Chatbot (Phi-3 + BM25, Fully Offline)

This project implements a fully local Retrieval-Augmented Generation (RAG) chatbot that runs offline on a laptop using a quantized Phi-3 Mini model and a BM25 retriever.
No internet, no API keys, no server — everything happens on your machine.

Features

Local Language Model using Phi-3 Mini (Q4 quantized or FP16 baseline)
Retrieval-Augmented Generation (RAG) using BM25 over a local text/PDF corpus
Fully Offline
CLI Chatbot (RAG mode + plain LLM mode)
Basic Guardrails (self-harm, violence, criminal instructions)
PDF/Text ingestion with chunking + scoring
Quantization Evaluation (FP16 vs Q4 comparison script)

Project Structure
RAG_chatbot/
  README.md
  rag_cli.py                # Main chatbot interface (REPL)
  setup.sh                  # Linux setup script

  rag/
    retriever.py           # BM25 retriever + PDF loader
    rag_engine.py          # RAG pipeline (LLM + retrieval)

  corpus/
    *.txt / *.pdf          # Your documents (contains some sample documents as well)

  models/
    Phi-3-mini-q4.gguf     # Quantized model
    Phi-3-mini-fp16.gguf   # Baseline model

  eval/                    # FP16 vs Q4 comparison scripts

Requirements

Python 3.9–3.12
Works on Windows, Linux, and macOS
No GPU required — optimized for CPU inference

Models

Phi-3-mini-q4.gguf (Quantized, fast)
Phi-3-mini-fp16.gguf (Baseline, slower)
Rename if needed or use CLI flags to point to a custom directory.

Download the Phi-3 Mini GGUF Models

Model weights cannot be included in this repository due to their size and licensing restrictions.
Please download them manually from Microsoft’s official HuggingFace page.

Create the models folder (if not created automatically):

mkdir -p models

Then download the recommended quantized model:

Recommended (fastest, lowest memory): Q4_K_M quantized model (~2.2 GB)
wget -O models/Phi-3-mini-q4.gguf \
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4_K_M.gguf

Optional (baseline comparison): FP16 full-precision model (~7 GB)
wget -O models/Phi-3-mini-fp16.gguf \
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf



Corpus Format

Place your documents inside:

corpus/

Supported formats:

.txt
.pdf (text PDFs, not scanned images)
Each document is automatically split into chunks, indexed, and used for retrieval.

Running the Chatbot

Activate your virtual environment, then:
python rag_cli.py

You will see:

=== Local RAG Chatbot ===
Model: models/Phi-3-mini-q4.gguf
Mode: RAG (BM25 + context)
Guardrails: ON
Commands: /exit, /quit, /help


CLI Options
Select FP16 vs Q4 model
python rag_cli.py --model fp16
python rag_cli.py --model q4

Disable RAG (plain LLM)
python rag_cli.py --no-rag

Customize generation
python rag_cli.py --max-tokens 512 --temperature 0.7

Change number of retrieved chunks
python rag_cli.py --k 5

Disable safety guardrails
python rag_cli.py --no-guardrails

Quantization Evaluation Summary

A small evaluation comparing general-knowledge question answering by the basemodel and the quantized model:

Num questions: 10

Q4:   passes = 6/10, avg recall ≈ 0.45, avg time ≈ 39s
FP16: passes = 7/10, avg recall ≈ 0.44, avg time ≈ 77s

Interpretation:

Quality is nearly identical
Q4 is about 2× faster
Therefore Q4 is the default model in this project.

System Architecture (High-Level)

1️ Corpus Layer
– Loads .txt and .pdf
– Extracts text → chunking → BM25 index

2️ Retrieval Layer (BM25)
– Scores chunks by relevance
– Returns top-k context pieces

3 Generation Layer (LLM)
– Builds a structured RAG prompt
– Phi-3 answers using only retrieved context
– CLI displays the answer + sources
