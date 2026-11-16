"""Index synthetic revisions and join structured measurements to text evidence."""

import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from math import sqrt
from urllib.request import Request, urlopen

ROWS = (
    ("d-old", "R42", "2026-11-02", "2026-11-10", "setup", "Warm-up lasted 20 minutes."),
    ("d-current", "R42", "2026-11-10", None, "setup", "Allow hardware to stabilize thermally before measurement."),
    ("d-decoy", "R42", "2026-11-10", None, "cleanup", "After the test, warm up the backup workstation."),
    ("d-other", "R41", "2026-11-10", None, "setup", "Warm up the hardware for 30 minutes before the test."),
    ("d-invalid", "R42", "2026-11-99", None, "setup", "Unverified date."),
)
DAY = "2026-11-16"
QUERY = "How should I warm up the hardware before the test?"


def ingest(db):
    rejected = []
    for record in ROWS:
        key, _, start, end, _, _ = record
        try:
            begin = date.fromisoformat(start)
            if end and date.fromisoformat(end) <= begin:
                raise ValueError("invalid validity interval")
        except ValueError:
            rejected.append(key)
            continue
        db.execute("INSERT INTO docs VALUES (?, ?, ?, ?, ?, ?)", record)
    return rejected


def embed(inputs):
    payload = json.dumps({"model": "all-minilm:22m", "input": inputs, "truncate": False}).encode()
    request = Request("http://127.0.0.1:11434/api/embed", data=payload,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=25) as response:
        return json.load(response)["embeddings"]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - {"a", "the", "i", "should", "to"}


def cosine(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True)) / (
        sqrt(sum(x * x for x in left)) * sqrt(sum(y * y for y in right))
    )


def rank(docs):
    question, *vectors = embed([QUERY, *(text for _, text in docs)])
    words = tokens(QUERY)
    lexical = sorted(((len(words & tokens(text)), key) for key, text in docs),
                     key=lambda row: (-row[0], row[1]))
    dense = sorted(((cosine(question, vector), key) for (key, _), vector in zip(docs, vectors, strict=True)),
                   key=lambda row: (-row[0], row[1]))
    hybrid = {}
    for results in (lexical, dense):
        for position, (_, key) in enumerate(results, 1):
            hybrid[key] = hybrid.get(key, 0) + 1 / (60 + position)
    return {"lexical": lexical, "dense": [(round(score, 4), key) for score, key in dense],
            "hybrid": sorted(hybrid, key=lambda key: (-hybrid[key], key))}


if __name__ == "__main__":
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE TABLE docs (id PRIMARY KEY, run, valid_from, valid_until, section, text)")
        db.execute("CREATE TABLE results (id PRIMARY KEY, detection_pct)")
        db.execute("INSERT INTO results VALUES ('R42', 84.0)")
        rejected = ingest(db)
        all_docs = db.execute("SELECT id,text FROM docs").fetchall()
        current = db.execute("""SELECT id,text FROM docs WHERE run=? AND valid_from<=?
            AND (valid_until IS NULL OR ?<valid_until)""", ("R42", DAY, DAY)).fetchall()
        qualified = db.execute("""SELECT d.id,d.text,r.detection_pct FROM docs d
            JOIN results r ON d.run=r.id WHERE r.id=? AND d.section='setup' AND d.valid_from<=?
            AND (d.valid_until IS NULL OR ?<d.valid_until)""", ("R42", DAY, DAY)).fetchall()
        print(json.dumps({"rejected": rejected, "all": rank(all_docs), "current": rank(current),
                          "verified_join": qualified}))
