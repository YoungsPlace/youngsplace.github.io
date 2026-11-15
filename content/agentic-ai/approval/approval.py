"""Check permission, exact confirmation and audit fields at a mock write boundary."""

from dataclasses import dataclass, replace
import json


@dataclass(frozen=True)
class Reservation:
    request_id: str
    actor: str
    room: str
    time: str


ALLOWED = {"u1": {"LAB-A", "LAB-B"}, "u2": {"LAB-B"}}
SLOTS = {"LAB-A": {"10:30"}, "LAB-B": {"14:00", "15:00"}}


def reserve(request, approved, committed, audit):
    if request.time not in SLOTS.get(request.room, set()):
        state = "invalid_slot"
    elif request.room not in ALLOWED.get(request.actor, set()):
        state = "denied"
    elif request != approved:
        state = "needs_matching_approval"
    elif request.request_id in committed:
        state = "same_request" if committed[request.request_id] == request else "request_conflict"
    else:
        committed[request.request_id] = request
        state = "committed"
    audit.append({"request_id": request.request_id, "tool": "reserve", "status": state})
    return state


def demo():
    committed, audit = {}, []
    ready = Reservation("r1", "u1", "LAB-B", "14:00")
    for label, request, approved, expected in [
        ("unapproved", ready, None, "needs_matching_approval"),
        ("changed_slot", replace(ready, time="15:00"), ready, "needs_matching_approval"),
        ("invalid_slot", replace(ready, time="25:00"), None, "invalid_slot"),
        ("wrong_scope", Reservation("r2", "u2", "LAB-A", "10:30"), None, "denied"),
        ("approved", ready, ready, "committed"),
        ("same_request", ready, ready, "same_request"),
        ("conflicting_request", replace(ready, time="15:00"), replace(ready, time="15:00"), "request_conflict"),
    ]:
        state = reserve(request, approved, committed, audit)
        assert state == expected, (label, state)
        print(label, state)
    assert committed == {"r1": ready}
    assert all(set(event) == {"request_id", "tool", "status"} for event in audit)
    print(json.dumps({"writes": len(committed), "events": audit}, indent=2))


if __name__ == "__main__":
    demo()
