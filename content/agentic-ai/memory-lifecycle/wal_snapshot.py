"""Compare a reader already in a WAL snapshot with a new reader after deletion."""

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from lifecycle import SCHEMA, change, search


def demo(path):
    with closing(sqlite3.connect(path)) as writer, closing(sqlite3.connect(path)) as reader:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.executescript(SCHEMA)
        change(writer, "u1", "preferred_lab", "cobalt")

        reader.execute("BEGIN")
        before = search(reader, "u1", "cobalt")
        change(writer, "u1", "preferred_lab", None)
        still_visible = search(reader, "u1", "cobalt")

        with closing(sqlite3.connect(path)) as fresh:
            fresh_result = search(fresh, "u1", "cobalt")

        error = None
        try:
            reader.execute("INSERT INTO changes(owner, name, action) VALUES (?, ?, ?)", ("u1", "preferred_lab", "put"))
        except sqlite3.OperationalError as exc:
            error = exc.sqlite_errorname
        finally:
            reader.rollback()
        after_reopen = search(reader, "u1", "cobalt")
        assert before == still_visible == ["cobalt"]
        assert fresh_result == after_reopen == []
        assert error == "SQLITE_BUSY_SNAPSHOT", error
        return {"before_delete": before, "old_snapshot_after_delete": still_visible,
                "fresh_reader": fresh_result, "old_reader_new_transaction": after_reopen,
                "upgrade_error": error}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(demo(Path(directory) / "wal.sqlite")))
