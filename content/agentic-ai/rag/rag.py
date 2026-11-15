"""Compare whole-document and section retrieval on synthetic lab manuals."""

import argparse
import json

from retrieve import cosine, embed, ollama

SECTIONS = (
    ("v1#book", False, "Submit a lab booking at least 24 hours before your visit."),
    ("v2#book", True, "Reserve shared lab equipment no later than 48 hours before arrival."),
    ("v2#access", True, "A campus identification card is required at the front desk."),
    ("v2#storage", True, "Place bags and coats in the numbered lockers near the entrance."),
    ("v2#cleanup", True, "Wipe the electronics bench and put tools away before leaving."),
    ("v2#transport", True, "The campus shuttle stops next to the main entrance."),
)
QUESTIONS = {
    "booking": "How many hours before my visit do I need to submit a lab booking?",
    "parking": "How much is the weekend parking fee?",
}


def ranked(question, records):
    query, *rows = embed([question, *(text for _, _, text in records)])
    scores = [(source, current, text, cosine(query, row))
              for (source, current, text), row in zip(records, rows, strict=True)]
    return sorted(scores, key=lambda item: (-item[-1], item[0]))


def generate(question, context):
    source, _, text, _ = context
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {"role": "system", "content": "Answer using the provided excerpt. If it does not contain the answer, reply NO_EVIDENCE. If it does, cite its ID in square brackets."},
            {"role": "user", "content": json.dumps({"question": question, "excerpt_id": source, "excerpt": text})},
        ],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 80},
    }
    return ollama("chat", payload)["message"]["content"].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", choices=QUESTIONS)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    question = QUESTIONS[args.question]
    documents = (
        ("v1", False, SECTIONS[0][2]),
        ("v2", True, " ".join(text for _, current, text in SECTIONS if current)),
    )
    whole = ranked(question, documents)
    chunked = ranked(question, SECTIONS)
    current = [row for row in chunked if row[1]]
    ranked_lists = [("whole", whole), ("sections", chunked), ("current", current)]
    for label, rows in ranked_lists:
        print(label, [(source, round(score, 4)) for source, _, _, score in rows])
    best = current[0]
    print("excerpt", best[0], best[2])
    if args.generate:
        print("answer", generate(question, best))
