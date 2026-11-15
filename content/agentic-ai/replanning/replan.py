"""Compare a fixed room plan with bounded recovery on synthetic read results."""

import argparse
import json

CASES = {
    "open": {"A": ("open", "10:30")},
    "closed": {"A": ("closed", None, "B"), "B": ("open", "14:00")},
    "empty": {"A": ("open", None, "B"), "B": ("open", "14:00")},
    "timeout_once": {"A": ("open", "10:30")},
    "timeout_twice": {"A": ("open", "10:30")},
    "cycle": {"A": ("closed", None, "B"), "B": ("closed", None, "A")},
    "no_slot": {"A": ("open", None)},
}
TIMEOUTS = {"timeout_once": 1, "timeout_twice": 2}


def search(case, adaptive, max_calls):
    current = "A"
    rooms = CASES[case]
    seen = {current}
    timeouts_left = TIMEOUTS.get(case, 0)
    trace = []

    def record(tool, result):
        trace.append({"tool": tool, "room": current, "result": result})

    def finish(status, slot=None):
        return {"status": status, "room": current, "slot": slot, "trace": trace}

    while len(trace) < max_calls:
        state, slot, *alternative = rooms[current]
        record("status", state)
        if state == "open":
            max_attempts = 2 if adaptive else 1
            for _ in range(max_attempts):
                if len(trace) == max_calls:
                    return finish("call_limit")
                if not timeouts_left:
                    record("first_slot", slot)
                    if slot is not None:
                        return finish("found", slot)
                    break
                timeouts_left -= 1
                record("first_slot", "timeout")
            else:
                return finish("timeout")
        if not adaptive or not alternative:
            return finish("no_slot" if state == "open" else "closed")
        next_room = alternative[0]
        if next_room in seen:
            return finish("cycle")
        seen.add(next_room)
        current = next_room
    return finish("call_limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fixed", "adaptive"])
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--max-calls", type=int, default=6)
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error("--max-calls must be at least 1")
    print(json.dumps(search(args.case, args.mode == "adaptive", args.max_calls), indent=2))
