"""Resume one synthetic investigation and separately store a confirmed preference."""

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from contextlib import closing


def open_db(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (owner, run, version, state, evidence, baseline,
            PRIMARY KEY(owner, run));
        CREATE TABLE IF NOT EXISTS preferences (owner PRIMARY KEY, unit NOT NULL);
    """)
    return db


def command(path, operation):
    owner, run = "team-a", "R42-investigation"
    with closing(open_db(path)) as db, db:
        if operation == "start":
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, 1, 'waiting', 'compare:R41:R42@2', NULL)",
                       (owner, run))
        elif operation == "resume":
            changed = db.execute("""UPDATE runs SET state='ready', version=2, baseline='R38'
                WHERE owner=? AND run=? AND version=1 AND state='waiting'""", (owner, run)).rowcount
            if not changed:
                return {"event": "resume", "status": "stale_or_duplicate"}
        elif operation == "remember":
            db.execute("INSERT OR REPLACE INTO preferences VALUES (?, 'percentage_points')", (owner,))
        row = db.execute("SELECT version,state,evidence,baseline FROM runs WHERE owner=? AND run=?",
                         (owner, run)).fetchone()
        pref = db.execute("SELECT unit FROM preferences WHERE owner=?", (owner,)).fetchone()
        return {"event": operation, "checkpoint": row, "preference": pref[0] if pref else None}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(command(sys.argv[1], sys.argv[2])))
    else:
        with TemporaryDirectory() as directory:
            db = Path(directory) / "resume.db"
            for event in ("start", "inspect", "resume", "resume", "remember", "inspect"):
                subprocess.run([sys.executable, __file__, str(db), event], check=True)
