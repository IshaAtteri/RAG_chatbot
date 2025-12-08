#!/usr/bin/env bash
set -e

# -------------------------------------------
# Local RAG Chatbot - Setup Script (Linux)
# -------------------------------------------
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# This will:
#   - Create a Python virtual environment in .venv
#   - Upgrade pip
#   - Install dependencies from requirements.txt
#   - Create the models/ and corpus/ folders if missing
#
# After running:
#   source .venv/bin/activate
#   python rag_cli.py --model q4
# -------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Using Python executable: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Could not find Python 3 (tried '$PYTHON_BIN')."
  echo "Please install Python 3.9+ and/or set PYTHON_BIN to the correct path."
  exit 1
fi

# ---------- Create virtual environment ----------

if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "[INFO] Virtual environment .venv already exists. Reusing it."
fi

VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[ERROR] Virtual environment Python not found at $VENV_PYTHON"
  exit 1
fi

echo "[INFO] Upgrading pip inside the virtual environment ..."
"$VENV_PIP" install --upgrade pip

# ---------- Install Python dependencies ----------

if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing dependencies from requirements.txt ..."
  "$VENV_PIP" install -r requirements.txt
else
  echo "[WARN] requirements.txt not found. Installing core packages directly ..."
  "$VENV_PIP" install "llama-cpp-python>=0.3.0" "rank-bm25>=0.2.2" "pypdf>=3.17.0" "numpy>=1.26.0"
fi

# ---------- Prepare folders ----------

echo "[INFO] Ensuring models/ and corpus/ directories exist ..."
mkdir -p models
mkdir -p corpus

# ---------- Model download instructions ----------

echo
echo "============================================="
echo "      MODEL FILES NOT INCLUDED BY GIT        "
echo "============================================="
echo "Due to size and licensing, GGUF model files"
echo "are NOT stored in this repository."
echo
echo "Your code expects the following files:"
echo "  models/Phi-3-mini-q4.gguf   (quantized, default)"
echo "  models/Phi-3-mini-fp16.gguf (optional baseline)"
echo
echo "Download Q4 model (recommended):"
echo "  wget -O models/Phi-3-mini-q4.gguf \\"
echo "    https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4_K_M.gguf"
echo
echo "Download FP16 model (optional, large ~7GB):"
echo "  wget -O models/Phi-3-mini-fp16.gguf \\"
echo "    https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf"
echo
echo "[INFO] Setup complete."
echo "To start using the environment, run:"
echo "  source .venv/bin/activate"
echo
echo "Then launch the chatbot with:"
echo "  python rag_cli.py --model q4"
echo "============================================="
