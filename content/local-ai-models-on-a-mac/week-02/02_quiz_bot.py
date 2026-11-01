import ollama

history = [
    {
        "role": "system",
        "content": (
            "You are a quiz master. Ask the user one trivia question "
            "at a time. After they answer, tell them if they're right "
            "or wrong and explain the correct answer. Then ask another "
            "question. Keep score."
        ),
    }
]

print("Quiz bot: type 'Start the quiz!' to begin.")
print("Type 'quit', 'exit', or 'bye' to exit.")

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
