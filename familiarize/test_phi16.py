from llama_cpp import Llama
import time

MODEL_PATH = "../models/Phi-3-mini-fp16.gguf"

print("Loading model...")
start = time.time()

llm = Llama(
    model_path = MODEL_PATH,
    n_ctx = 2048,
    n_threads = 8,
)

print(f"Model loaded in {time.time() - start:.2f} seconds.\n")

# Use the model's actual chat template through create_chat_completion
messages = [
    {"role": "user", "content": "Explain what a neural network is in two sentences."}
]

print("Generating text...")
start_gen = time.time()

output = llm.create_chat_completion(
    messages=messages,
    max_tokens=150,
    temperature=0.7,
    top_p=0.9,
)

print(f"Generation took {time.time() - start_gen:.2f} seconds.\n")

print("=== Output ===")
print(output["choices"][0]["message"]["content"].strip())
