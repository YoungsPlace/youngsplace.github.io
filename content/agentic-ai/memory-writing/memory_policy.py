"""Classify synthetic observations for durable, session-only, or discarded storage."""

import json
from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class Observation:
    key: str
    value: str
    source: str
    scope: str
    recorded_on: str
    valid_from: str
    valid_until: str | None = None


ROWS = (
    Observation("preferred_lab", "LAB-A", "u1:t1:3", "user_confirmed", "2026-11-16", "2026-11-16"),
    Observation("preferred_lab", "LAB-B", "u1:t1:7", "user_confirmed", "2026-11-16", "2026-12-01"),
    Observation("room_status", "LAB-A closed", "lookup:r17", "tool_observation", "2026-11-16", "2026-11-16", "2026-11-17"),
    Observation("preferred_lab", "LAB-C", "u1:t1:8", "inferred", "2026-11-16", "2026-11-16"),
    Observation("preferred_lab", "LAB-B", "u1:t1:9", "this_request", "2026-11-16", "2026-11-16"),
)


def classify(row):
    if row.key == "preferred_lab" and row.scope == "user_confirmed":
        return "durable"
    if row.scope in {"tool_observation", "this_request"}:
        return "session"
    return "discard"


def preference_on(rows, day, as_known_on):
    when = date.fromisoformat(day)
    since = date.fromisoformat(as_known_on)
    eligible = [
        row for row in rows
        if classify(row) == "durable"
        and date.fromisoformat(row.recorded_on) <= since
        and date.fromisoformat(row.valid_from) <= when
        and (row.valid_until is None or when < date.fromisoformat(row.valid_until))
    ]
    return max(eligible, key=lambda row: (row.valid_from, row.recorded_on), default=None)


if __name__ == "__main__":
    for row in ROWS:
        print(json.dumps({"route": classify(row), **asdict(row)}))
    for day, known in [("2026-11-30", "2026-11-16"), ("2026-12-01", "2026-11-16"), ("2026-12-01", "2026-11-15")]:
        row = preference_on(ROWS, day, known)
        print(json.dumps({"day": day, "as_known_on": known, "value": row.value if row else None}))
