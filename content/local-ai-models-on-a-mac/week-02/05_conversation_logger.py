import argparse
from pathlib import Path

import ollama

parser = argparse.ArgumentParser(description="Chat and save the conversation on exit.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).with_name("chat_log.txt"),
)
args = parser.parse_args()

history = [
    {
        "role": "system",
        "content": (
            "You are a friendly and helpful assistant. "
            "Be concise but thorough in your answers."
        ),
    }
]

print("Chat with AI! (type 'quit', 'exit', or 'bye' to save and exit)")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "bye"):
        print("\nGoodbye!")
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})
    stream = ollama.chat(
        model="gemma3:4b",
        messages=history,
        stream=True,
    )

    print("\nAI: ", end="", flush=True)
    response_text = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        response_text += piece
    print()

    history.append({"role": "assistant", "content": response_text})

with args.output.open("w", encoding="utf-8") as file:
    for message in history:
        file.write(f"[{message['role'].upper()}]\n{message['content']}\n\n")

print(f"Conversation saved to {args.output}")
