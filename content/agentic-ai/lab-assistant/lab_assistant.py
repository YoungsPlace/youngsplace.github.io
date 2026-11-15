"""Run a read-only lab assistant and a mock, confirmed reservation workflow."""

import argparse
import json
from dataclasses import dataclass

from approval import ALLOWED, Reservation, reserve

PUBLIC_POLICIES = (
    {"source": "v1#book", "current": False, "notice_hours": 24},
    {"source": "v2#book", "current": True, "notice_hours": 48},
)
PREFERENCES = {"u1": "LAB-B"}
ROOMS = {"LAB-A": "10:30", "LAB-B": "14:00"}


@dataclass(frozen=True)
class Request:
    id: str
    owner: str
    intent: str
    notice_hours: int = 0


def baseline(request, available):
    if request.intent != "read":
        return "unhandled"
    if "LAB-A" not in ALLOWED.get(request.owner, ()):
        return "no_accessible_slot"
    return f"LAB-A {available['LAB-A']}" if "LAB-A" in available else "no_slot"


def current_policy(retrieval):
    current, = (row for row in PUBLIC_POLICIES if row["current"])
    if not retrieval:
        return current

    from rag import QUESTIONS, SECTIONS, ranked

    candidates = [row for row in SECTIONS if row[1]]
    source, _, _, score = ranked(QUESTIONS["booking"], candidates)[0]
    if source != current["source"]:
        raise ValueError("the retrieved section has no verified booking policy")
    print(json.dumps({"retrieved_policy": source, "cosine": round(score, 4)}))
    return current


def assist(request, available, approved, writes, audit, policy):
    if request.intent not in {"read", "reserve"}:
        return {"status": "unsupported_intent", "source": None}
    preferred = PREFERENCES.get(request.owner)
    candidates = dict.fromkeys((preferred, "LAB-A", "LAB-B"))
    permitted = ALLOWED.get(request.owner, ())
    room = next((candidate for candidate in candidates
                 if candidate in permitted and candidate in available), None)
    if room is None:
        return {"status": "no_accessible_slot", "source": None}
    time = available[room]
    result = {"room": room, "time": time, "source": policy["source"], "notice_hours": policy["notice_hours"]}
    if request.intent == "read":
        return {"status": "found", **result}
    if request.notice_hours < policy["notice_hours"]:
        return {"status": "notice_too_short", **result}
    proposal = Reservation(request.id, request.owner, room, time)
    return {"status": reserve(proposal, approved, writes, audit), **result}


def demo(policy):
    writes, audit = {}, []
    approved_b = Reservation("r6", "u1", "LAB-B", "14:00")
    cases = [
        ("known_preference", Request("r1", "u1", "read"), ROOMS, None, "found", "LAB-B"),
        ("preferred_closed", Request("r2", "u1", "read"), {"LAB-A": "10:30"}, None, "found", "LAB-A"),
        ("other_user", Request("r3", "u2", "read"), {"LAB-A": "10:30"}, None, "no_accessible_slot", None),
        ("too_late", Request("r4", "u1", "reserve", 30), ROOMS, None, "notice_too_short", "LAB-B"),
        ("await_confirmation", Request("r5", "u1", "reserve", 48), ROOMS, None, "needs_matching_approval", "LAB-B"),
        ("confirmed", Request("r6", "u1", "reserve", 48), ROOMS, approved_b, "committed", "LAB-B"),
        ("duplicate", Request("r6", "u1", "reserve", 48), ROOMS, approved_b, "same_request", "LAB-B"),
    ]
    for label, request, available, approved, status, room in cases:
        result = assist(request, available, approved, writes, audit, policy)
        assert (result["status"], result.get("room")) == (status, room), result
        assert result["source"] == ("v2#book" if room else None), result
        print(json.dumps({"case": label, "baseline": baseline(request, available), "workflow": result}))
    assert writes == {"r6": approved_b}
    assert len(audit) == 3
    print(json.dumps({"writes": len(writes), "write_boundary_events": len(audit)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieve-policy", action="store_true")
    args = parser.parse_args()
    demo(current_policy(args.retrieve_policy))
