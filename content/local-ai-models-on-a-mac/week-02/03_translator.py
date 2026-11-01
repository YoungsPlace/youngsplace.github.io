import ollama

SYSTEM_MESSAGE = (
    "You are a translator. When the user gives you text, "
    "translate it into Japanese, Spanish, and French. "
    "Format your response as a clean list."
)

print("Japanese / Spanish / French translator")
print("Type 'quit', 'exit', or 'bye' to exit.")

while True:
    text = input("\nText: ").strip()
    if text.lower() in ("quit", "exit", "bye"):
        print("\nGoodbye!")
        break
    if not text:
        continue

    stream = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": text},
        ],
        stream=True,
    )

    print()
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()
