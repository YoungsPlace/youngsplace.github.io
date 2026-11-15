"""Rank synthetic memory entries with word overlap and local text embeddings."""

import argparse
import json
from collections import defaultdict
import re
from math import sqrt
from urllib.request import Request, urlopen

MODEL = "all-minilm:22m"
ROWS = (
    {"id": "m1", "owner": "u1", "start": "2026-10-01", "end": None,
     "text": "The lab hosts a board game club every Tuesday."},
    {"id": "m2", "owner": "u1", "start": "2026-10-01", "end": "2026-11-01",
     "text": "Use the electronics bench in LAB-A for soldering printed circuit boards."},
    {"id": "m3", "owner": "u1", "start": "2026-11-01", "end": None,
     "text": "Use the electronics bench in LAB-B for soldering printed circuit boards."},
    {"id": "m4", "owner": "u2", "start": "2026-11-01", "end": None,
     "text": "Where can I solder a board? Use the private electronics bench in LAB-C."},
)
QUERY = "Where can I solder a board?"


def ollama(endpoint, payload):
    request = Request(
        f"http://127.0.0.1:11434/api/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def embed(texts):
    return ollama("embed", {"model": MODEL, "input": texts, "truncate": False})["embeddings"]


def tokens(text):
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


def cosine(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True)) / (
        sqrt(sum(x * x for x in left)) * sqrt(sum(x * x for x in right))
    )


def rank(ids, scores):
    return sorted(zip(ids, scores, strict=True), key=lambda item: (-item[1], item[0]))


def eligible(rows, owner, day):
    return [row for row in rows if row["owner"] == owner and row["start"] <= day
            and (row["end"] is None or day < row["end"])]


def compare(query, rows):
    if not rows:
        return {"lexical": [], "dense": []}
    ids = [row["id"] for row in rows]
    q_tokens = tokens(query)
    query_vector, *vectors = embed([query, *(row["text"] for row in rows)])
    dense = rank(ids, [cosine(query_vector, vector) for vector in vectors])
    return {
        "lexical": rank(ids, [len(q_tokens & tokens(row["text"])) for row in rows]),
        "dense": [(item_id, round(score, 4)) for item_id, score in dense],
        "dimensions": len(query_vector),
    }


def reciprocal_rank_fusion(*lists, k=60):
    scores = defaultdict(float)
    for items in lists:
        for position, (item_id, _) in enumerate(items, 1):
            scores[item_id] += 1 / (k + position)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    if args.hybrid:
        rows = eligible(ROWS, "u1", "2026-11-16")
        for query in (QUERY, "Where can I solder a printed circuit board?"):
            result = compare(query, rows)
            lexical_hits = [row for row in result["lexical"] if row[1] > 0]
            fused = reciprocal_rank_fusion(lexical_hits, result["dense"])
            print(json.dumps({"query": query, **result, "rrf": fused}))
        raise SystemExit(0)
    for label, rows in [
        ("unfiltered_demo", ROWS),
        ("current_u1", eligible(ROWS, "u1", "2026-11-16")),
        ("previous_u1", eligible(ROWS, "u1", "2026-10-15")),
        ("unknown_user", eligible(ROWS, "u3", "2026-11-16")),
    ]:
        print(json.dumps({"scope": label, **compare(QUERY, rows)}))
