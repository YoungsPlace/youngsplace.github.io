"""Read a synthetic preference on independent valid-time and record-time axes."""

import json
import sqlite3
from contextlib import closing

ROWS = [
    ("u1", "preferred_lab", "LAB-A", "2026-11-16", "2026-12-01", "2026-11-16", None),
    ("u1", "preferred_lab", "LAB-B", "2026-12-01", None, "2026-11-16", "2026-12-03"),
    ("u1", "preferred_lab", "LAB-C", "2026-12-01", None, "2026-12-03", None),
]
SQL = """
SELECT value FROM memory
WHERE owner = ? AND name = ?
  AND valid_from <= ? AND (valid_until IS NULL OR ? < valid_until)
  AND known_from <= ? AND (known_until IS NULL OR ? < known_until)
ORDER BY value
"""


def read_as_of(db, owner, key, day, known):
    values = [value for (value,) in db.execute(SQL, (owner, key, day, day, known, known))]
    if len(values) > 1:
        return {"status": "conflict", "values": values}
    return {"status": "found", "value": values[0]} if values else {"status": "no_evidence"}


if __name__ == "__main__":
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("""CREATE TABLE memory (owner, name, value,
                      valid_from, valid_until, known_from, known_until)""")
        db.executemany("INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?, ?)", ROWS)
        for day, known in [
            ("2026-11-30", "2026-12-04"),
            ("2026-12-02", "2026-12-02"),
            ("2026-12-02", "2026-12-04"),
            ("2026-12-03", "2026-12-03"),
        ]:
            print(json.dumps({"day": day, "known": known, **read_as_of(db, "u1", "preferred_lab", day, known)}))
        db.execute("INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?, ?)",
                   ("u1", "preferred_lab", "LAB-D", "2026-12-01", None, "2026-12-03", None))
        conflict = read_as_of(db, "u1", "preferred_lab", "2026-12-02", "2026-12-04")
        assert conflict == {"status": "conflict", "values": ["LAB-C", "LAB-D"]}
        print(json.dumps(conflict))
