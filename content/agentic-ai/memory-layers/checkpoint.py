"""Inspect a synthetic thread checkpoint and its owner's saved preference."""

import argparse
import json
import sqlite3
from contextlib import closing


def run(path, action, owner, thread):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS checkpoints (owner, thread, payload, PRIMARY KEY (owner, thread))")
        db.execute("CREATE TABLE IF NOT EXISTS preferences (owner PRIMARY KEY, preferred_lab)")
        if action == "pause":
            db.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?)", (owner, "LAB-A"))
            state = {"observed": {"LAB-A": "closed", "alternate": "LAB-B"}, "next": "room_status", "room": "LAB-B"}
            db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)", (owner, thread, json.dumps(state)))
        checkpoint = db.execute("SELECT payload FROM checkpoints WHERE owner=? AND thread=?", (owner, thread)).fetchone()
        preference = db.execute("SELECT preferred_lab FROM preferences WHERE owner=?", (owner,)).fetchone()
        return {
            "thread": thread,
            "checkpoint": json.loads(checkpoint[0]) if checkpoint else None,
            "preferred_lab": preference[0] if preference else None,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pause", "inspect"])
    parser.add_argument("db")
    parser.add_argument("owner")
    parser.add_argument("thread")
    args = parser.parse_args()
    print(json.dumps(run(args.db, args.action, args.owner, args.thread)))
