"""Exercise correction, deletion, source scoping and rollback in a temporary database."""

import json
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE memory (owner TEXT, name TEXT, value TEXT NOT NULL,
    PRIMARY KEY (owner, name));
CREATE VIRTUAL TABLE memory_search USING fts5(owner UNINDEXED, name UNINDEXED, value);
CREATE TABLE changes (event_id INTEGER PRIMARY KEY, owner TEXT, name TEXT,
    action TEXT CHECK(action IN ('put', 'delete')));
"""


def change(db, owner, name, value, *, interrupt=False):
    with db:
        db.execute("DELETE FROM memory_search WHERE owner=? AND name=?", (owner, name))
        if value is None:
            db.execute("DELETE FROM memory WHERE owner=? AND name=?", (owner, name))
        else:
            db.execute("INSERT INTO memory VALUES (?, ?, ?) ON CONFLICT(owner, name) DO UPDATE SET value=excluded.value",
                       (owner, name, value))
            db.execute("INSERT INTO memory_search VALUES (?, ?, ?)", (owner, name, value))
        if interrupt:
            raise RuntimeError("simulated failure before commit")
        db.execute("INSERT INTO changes(owner, name, action) VALUES (?, ?, ?)",
                   (owner, name, "delete" if value is None else "put"))


def search(db, owner, word):
    if not re.fullmatch(r"[a-zA-Z]+", word):
        raise ValueError("expected one alphabetic search term")
    rows = db.execute("""SELECT m.value
        FROM memory_search AS idx
        JOIN memory AS m ON m.owner=idx.owner AND m.name=idx.name
        WHERE idx.owner=? AND memory_search MATCH ?""", (owner, word))
    return [value for (value,) in rows]


def answer(db, owner, word):
    matches = search(db, owner, word)
    return matches[0] if matches else "no_evidence"


def demo(path):
    with closing(sqlite3.connect(path)) as db:
        db.executescript(SCHEMA)
        change(db, "u1", "preferred_lab", "amber")
        change(db, "u2", "preferred_lab", "cedar")
        change(db, "u1", "preferred_lab", "cobalt")
        assert search(db, "u1", "amber") == []
        print("corrected", answer(db, "u1", "cobalt"))
        try:
            change(db, "u1", "preferred_lab", "silver", interrupt=True)
        except RuntimeError:
            assert search(db, "u1", "silver") == []
            assert answer(db, "u1", "cobalt") == "cobalt"
            print("rollback", answer(db, "u1", "cobalt"))
        assert answer(db, "u1", "cedar") == "no_evidence"
        print("different_owner", answer(db, "u1", "cedar"))
        change(db, "u1", "preferred_lab", None)

    with closing(sqlite3.connect(path)) as db:
        actual = {word: answer(db, "u1", word) for word in ("amber", "cobalt", "silver")}
        assert actual == {"amber": "no_evidence", "cobalt": "no_evidence", "silver": "no_evidence"}
        assert answer(db, "u2", "cedar") == "cedar"
        events = [action for (action,) in db.execute("SELECT action FROM changes WHERE owner=? ORDER BY event_id", ("u1",))]
        assert events == ["put", "put", "delete"]
        print("reopened", json.dumps(actual), "u2", answer(db, "u2", "cedar"), "events", events)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        demo(Path(directory) / "study.sqlite")
