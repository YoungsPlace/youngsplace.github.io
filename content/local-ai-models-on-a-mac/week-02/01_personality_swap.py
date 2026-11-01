import ollama

MODEL = "gemma3:4b"
QUESTION = "What is photosynthesis?"

PERSONALITIES = {
    "Strict math professor": (
        "You are a strict math professor who demands precision."
    ),
    "Enthusiastic kindergarten teacher": (
        "You are an enthusiastic kindergarten teacher who uses simple words "
        "and lots of encouragement."
    ),
    "Grumpy librarian": (
        "You are a grumpy but knowledgeable librarian who always recommends a book."
    ),
}

for name, system_message in PERSONALITIES.items():
    print(f"\n{'=' * 12} {name} {'=' * 12}")
    print(f"Question: {QUESTION}\n")

    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": QUESTION},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()
