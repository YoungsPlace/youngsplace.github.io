import argparse
import time

import ollama

parser = argparse.ArgumentParser(description="Measure a streaming chat response.")
parser.add_argument("model", nargs="?", default="gemma3:4b")
args = parser.parse_args()

QUESTION = "Explain how the internet works in simple terms."
print(f"Model: {args.model}")
print(f"Question: {QUESTION}\n")

start = time.perf_counter()
first_text_time = None
last_chunk = None

stream = ollama.chat(
    model=args.model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide detailed, thoughtful answers.",
        },
        {"role": "user", "content": QUESTION},
    ],
    stream=True,
)

for chunk in stream:
    last_chunk = chunk
    piece = chunk["message"]["content"]
    if piece:
        print(piece, end="", flush=True)
        if first_text_time is None:
            first_text_time = time.perf_counter() - start

elapsed = time.perf_counter() - start
print("\n")
if first_text_time is not None:
    print(f"First text displayed: {first_text_time:.3f} seconds")
print(f"Total response time: {elapsed:.3f} seconds")

load_duration = getattr(last_chunk, "load_duration", None)
eval_count = getattr(last_chunk, "eval_count", None)
if load_duration is not None:
    print(f"Model loading time: {load_duration / 1_000_000_000:.3f} seconds")
if eval_count is not None:
    print(f"Output tokens: {eval_count}")
