from llama_cpp import Llama
import time

MODEL_PATH = "../models/Phi-3-mini-q4.gguf"

print("Loading model...")
start = time.time()

llm = Llama(
    model_path = MODEL_PATH,
    n_ctx = 2048,
    n_threads = 8,
)

print(f"Model loaded in {time.time() - start:.2f} seconds.\n")

prompt = "Explain what a neural network is in two sentences."

print("Generating text...")
start_gen = time.time()

output = llm(prompt, max_tokens=150)

print(f"Generation took {time.time() - start_gen:.2f} seconds.\n")

print("=== Output ===")
print(output["choices"][0]["text"].strip())

