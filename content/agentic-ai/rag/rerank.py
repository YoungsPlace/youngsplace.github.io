"""Compare exact vector retrieval with a local query–passage cross-encoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from rag import QUESTIONS, SECTIONS, ranked

TESTS = {
    "booking": (
        QUESTIONS["booking"],
        "v2#book",
        [*SECTIONS, ("v2#checkin", True, "For lab booking, arrive 10 minutes early and check in at the door.")],
    ),
    "equipment": (
        "What is the minimum notice for borrowing lab equipment?",
        "v2#book",
        [*SECTIONS, ("v2#returns", True, "Return borrowed lab equipment within 24 hours after checkout.")],
    ),
    "parking": (QUESTIONS["parking"], None, SECTIONS),
}


def load_cross_encoder(directory):
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    return tokenizer, session


def rerank(query, candidates, tokenizer, session):
    encoded = tokenizer.encode_batch([(query, text) for _, _, text, _ in candidates])
    arrays = {
        "input_ids": [item.ids for item in encoded],
        "attention_mask": [item.attention_mask for item in encoded],
        "token_type_ids": [item.type_ids for item in encoded],
    }
    inputs = {item.name: np.asarray(arrays[item.name], dtype=np.int64) for item in session.get_inputs()}
    logits = session.run(None, inputs)[0].reshape(-1)
    assigned = [(row[0], float(score)) for row, score in zip(candidates, logits, strict=True)]
    return sorted(assigned, key=lambda item: (-item[1], item[0]))


def show(rows):
    return [[source, round(score, 4)] for source, score in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    tokenizer, session = load_cross_encoder(args.model_directory)
    for case, (query, gold, records) in TESTS.items():
        dense = ranked(query, [record for record in records if record[1]])
        shortlist = dense[:args.top_k]
        print(json.dumps({"case": case, "gold": gold,
                          "dense": show([(source, score) for source, _, _, score in shortlist]),
                          "cross": show(rerank(query, shortlist, tokenizer, session))}))
