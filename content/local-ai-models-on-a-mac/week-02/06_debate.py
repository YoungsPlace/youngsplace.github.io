import ollama

MODEL_A = "gemma3:4b"
MODEL_B = "phi3:mini"
ROUNDS = 3

history_a = [
    {
        "role": "system",
        "content": "You believe cats are better pets than dogs. Argue your position.",
    }
]
history_b = [
    {
        "role": "system",
        "content": "You believe dogs are better pets than cats. Argue your position.",
    }
]


def speak(model, history, label):
    print(f"\n[{label} / {model}]")
    stream = ollama.chat(model=model, messages=history, stream=True)
    reply = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        reply += piece
    print()
    return reply


message = "Why are cats the best pets?"
print("Debate: cats vs dogs")

for round_number in range(ROUNDS):
    print(f"\n{'=' * 12} Round {round_number + 1} {'=' * 12}")

    history_a.append({"role": "user", "content": message})
    reply_a = speak(MODEL_A, history_a, "A: cats")
    history_a.append({"role": "assistant", "content": reply_a})

    history_b.append({"role": "user", "content": reply_a})
    reply_b = speak(MODEL_B, history_b, "B: dogs")
    history_b.append({"role": "assistant", "content": reply_b})

    message = reply_b

print("\nDebate finished.")
