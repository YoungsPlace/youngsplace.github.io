"""Compare three designs on read-only, synthetic lab-availability data."""

import argparse
import json
from functools import partial
from urllib.error import HTTPError
from urllib.request import Request, urlopen


CASES = {
    "regular": {"LAB-A": {"open": True, "slot": "10:30"}},
    "fallback": {
        "LAB-A": {"open": False, "alternate": "LAB-B"},
        "LAB-B": {"open": True, "slot": "14:00"},
    },
    "empty": {"LAB-A": {"open": True, "slot": None}},
    "cycle": {
        "LAB-A": {"open": False, "alternate": "LAB-B"},
        "LAB-B": {"open": False, "alternate": "LAB-A"},
    },
}
ROOM_IDS = ("LAB-A", "LAB-B")
ACTIONS = ("room_status", "first_slot", "finish")

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "room": {"type": "string", "enum": list(ROOM_IDS)},
    },
    "required": ["action", "room"],
    "additionalProperties": False,
}

SYSTEM = """Find the first available time in primary room LAB-A.
Select exactly one next action and return JSON with the keys action and room.
Use these rules based on the last observation:
- With no observations, check LAB-A using room_status.
- If room_status returns open, call first_slot for that same room.
- If room_status returns closed with an alternate, call room_status for
  that alternate first. Its status is still unknown.
- Only after first_slot returns, use finish for the same room.
  A null slot means there is no available time and still allows finish.
Python executes and verifies each proposed action.
"""


def room_status(data, room):
    entry = data.get(room)
    if entry is None:
        return {"status": "unknown"}
    if entry["open"]:
        return {"status": "open"}
    return {"status": "closed", "alternate": entry.get("alternate")}


def first_slot(data, room):
    return {"slot": data[room]["slot"]}


def outcome(state, trace, room=None, slot=None, *, http_status=None):
    result = {"status": state, "room": room, "slot": slot, "trace": trace}
    if http_status is not None:
        result["http_status"] = http_status
    return result


def rejected(action, room, reason, trace):
    trace.append({"action": action, "room": room, "rejected": reason})
    return outcome(reason, trace)


def from_slot(trace, room, slot):
    return outcome("found" if slot else "no_slot", trace, room, slot)


def fixed_workflow(data):
    """The application owns the initial check and one explicit fallback."""
    trace = []
    room = "LAB-A"
    status = room_status(data, room)
    trace.append({"action": "room_status", "room": room, "result": status})
    if status.get("alternate") in ROOM_IDS:
        room = status["alternate"]
        status = room_status(data, room)
        trace.append({"action": "room_status", "room": room, "result": status})
    if status["status"] != "open":
        return outcome("unavailable", trace)
    result = first_slot(data, room)
    trace.append({"action": "first_slot", "room": room, "result": result})
    return from_slot(trace, room, result["slot"])


def scripted_choice(trace):
    """Deterministic stand-in for a model, used to test the runtime."""
    if not trace:
        return {"action": "room_status", "room": "LAB-A"}
    last = trace[-1]
    if last["action"] == "first_slot":
        return {"action": "finish", "room": last["room"]}
    if last["result"]["status"] == "open":
        return {"action": "first_slot", "room": last["room"]}
    return {"action": "room_status", "room": last["result"].get("alternate", last["room"])}


def ollama_choice(trace, *, model, schema=SCHEMA):
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"observations": trace})},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 80},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        reply = json.load(response)
    return json.loads(reply["message"]["content"])


def masked_choice(trace, *, model):
    """Constrain each field to the only move prescribed by this tiny scenario."""
    allowed = scripted_choice(trace)
    schema = {
        **SCHEMA,
        "properties": {
            key: {"type": "string", "enum": [value]}
            for key, value in allowed.items()
        },
    }
    return ollama_choice(trace, model=model, schema=schema)


def run_agent(data, choose, max_steps=5):
    observations = {}
    available_rooms = {"LAB-A"}
    trace = []
    tools = {"room_status": room_status, "first_slot": first_slot}
    for _ in range(max_steps):
        try:
            choice = choose(trace)
        except HTTPError as error:
            return outcome("decision_error", trace, http_status=error.code)
        except (OSError, ValueError, KeyError, TypeError):
            return outcome("decision_error", trace)
        if not isinstance(choice, dict) or set(choice) != {"action", "room"}:
            return outcome("invalid_action", trace)
        action, room = choice["action"], choice["room"]
        if action not in ACTIONS or room not in ROOM_IDS:
            return outcome("invalid_action", trace)
        if action == "finish":
            result = observations.get(("first_slot", room))
            if result is None:
                return rejected(action, room, "missing_evidence", trace)
            trace.append({"action": "finish", "room": room})
            return from_slot(trace, room, result["slot"])
        if room not in available_rooms:
            return rejected(action, room, "unobserved_room", trace)
        if action == "first_slot" and observations.get(("room_status", room), {}).get("status") != "open":
            return rejected(action, room, "room_not_confirmed_open", trace)

        key = (action, room)
        if key in observations:
            return rejected(action, room, "repeat", trace)
        result = tools[action](data, room)
        observations[key] = result
        trace.append({"action": action, "room": room, "result": result})
        if action == "room_status":
            alternate = result.get("alternate")
            if alternate in ROOM_IDS:
                available_rooms.add(alternate)
            if result["status"] != "open" and alternate not in ROOM_IDS:
                return outcome("unavailable", trace)
    return outcome("step_limit", trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("chat", "workflow", "scripted", "ollama", "masked"))
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    data = CASES[args.case]
    if args.mode == "chat":
        result = outcome("no_live_data", [])
    elif args.mode == "workflow":
        result = fixed_workflow(data)
    else:
        choose = {
            "scripted": scripted_choice,
            "ollama": partial(ollama_choice, model=args.model),
            "masked": partial(masked_choice, model=args.model),
        }[args.mode]
        result = run_agent(data, choose, args.max_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
