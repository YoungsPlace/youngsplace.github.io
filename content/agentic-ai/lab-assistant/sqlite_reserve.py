"""Exercise atomic mock reservations, replay and an outbox in a temporary SQLite file."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

SCHEMA = """
CREATE TABLE policy (version INTEGER PRIMARY KEY);
INSERT INTO policy VALUES (2);
CREATE TABLE approvals (
    actor TEXT NOT NULL, request_id TEXT NOT NULL, room TEXT NOT NULL, slot TEXT NOT NULL,
    policy_version INTEGER NOT NULL, expires INTEGER NOT NULL,
    PRIMARY KEY (actor, request_id)
);
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY, actor TEXT NOT NULL, request_id TEXT NOT NULL,
    room TEXT NOT NULL, slot TEXT NOT NULL, UNIQUE (actor, request_id), UNIQUE (room, slot)
);
CREATE TABLE outbox (reservation_id INTEGER PRIMARY KEY, event_type TEXT NOT NULL);
"""


class SimulatedCrash(RuntimeError):
    pass


def reserve(path, actor, request_id, room, slot, now, *, fail_before_commit=False):
    """The service supplies the authenticated actor and its current clock."""
    with closing(sqlite3.connect(path, timeout=5.0)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT id, room, slot FROM reservations WHERE actor=? AND request_id=?",
                           (actor, request_id)).fetchone()
        if prior:
            if prior[1:] != (room, slot):
                return {"status": "request_conflict"}
            return {"status": "already_committed", "reservation_id": prior[0]}

        approved = db.execute("""SELECT a.room, a.slot, a.expires
            FROM approvals AS a JOIN policy AS p ON p.version = a.policy_version
            WHERE actor=? AND request_id=?""", (actor, request_id)).fetchone()
        if approved is None or approved[:2] != (room, slot):
            return {"status": "needs_approval"}
        if now >= approved[2]:
            return {"status": "approval_expired"}

        inserted = db.execute("""INSERT INTO reservations(actor, request_id, room, slot)
            VALUES (?, ?, ?, ?) ON CONFLICT(room, slot) DO NOTHING""",
                              (actor, request_id, room, slot))
        if inserted.rowcount != 1:
            return {"status": "slot_taken"}
        reservation_id = inserted.lastrowid
        db.execute("INSERT INTO outbox VALUES (?, ?)", (reservation_id, "reserved"))
        if fail_before_commit:
            raise SimulatedCrash("after writing reservation and event")
        return {"status": "committed", "reservation_id": reservation_id}


def counts(path):
    with closing(sqlite3.connect(path)) as db:
        return [db.execute("SELECT count(*) FROM reservations").fetchone()[0],
                db.execute("SELECT count(*) FROM outbox").fetchone()[0]]


def demo(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?)", [
            ("u1", "r1", "LAB-B", "14:00", 2, 60),
            ("u2", "r2", "LAB-B", "14:00", 2, 60),
            ("u1", "r3", "LAB-A", "10:30", 2, 60),
            ("u1", "r4", "LAB-A", "10:30", 1, 60),
        ])
    barrier = Barrier(2)

    def compete(actor, request_id):
        barrier.wait(timeout=5)
        return actor, request_id, reserve(path, actor, request_id, "LAB-B", "14:00", 25)

    with ThreadPoolExecutor(max_workers=2) as pool:
        contenders = [pool.submit(compete, *identity) for identity in [("u1", "r1"), ("u2", "r2")]]
        results = [future.result() for future in contenders]
    assert sorted(result["status"] for _, _, result in results) == ["committed", "slot_taken"]
    actor, request_id, committed = next(row for row in results if row[-1]["status"] == "committed")
    replay = reserve(path, actor, request_id, "LAB-B", "14:00", 61)
    assert replay == {"status": "already_committed", "reservation_id": committed["reservation_id"]}
    changed = reserve(path, actor, request_id, "LAB-B", "15:00", 26)
    expired = reserve(path, "u1", "r3", "LAB-A", "10:30", 60)
    stale = reserve(path, "u1", "r4", "LAB-A", "10:30", 25)
    assert [changed["status"], expired["status"], stale["status"]] == [
        "request_conflict", "approval_expired", "needs_approval"
    ]

    try:
        reserve(path, "u1", "r3", "LAB-A", "10:30", 25, fail_before_commit=True)
    except SimulatedCrash:
        after_rollback = counts(path)
    else:
        raise AssertionError("expected simulated failure")
    assert after_rollback == [1, 1]
    recovered = reserve(path, "u1", "r3", "LAB-A", "10:30", 26)
    after_recovery = counts(path)
    assert recovered["status"] == "committed" and after_recovery == [2, 2]
    print(json.dumps({"race": [row[-1]["status"] for row in results], "replay": replay, "changed": changed,
                      "expired": expired, "stale_policy": stale, "after_rollback": after_rollback,
                      "after_recovery": after_recovery}, indent=2))


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        demo(Path(directory) / "reservations.sqlite")
