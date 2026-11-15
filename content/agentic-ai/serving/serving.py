"""Produce deterministic traces for a serving rollout with simulated time."""

import argparse
import json

SCENARIOS = {
    "normal": {"queue": 8, "model": 30, "tool": [(15, "ok")]},
    "transient": {"queue": 8, "model": 30, "tool": [(30, "timeout"), (15, "ok")]},
    "slow_model": {"queue": 8, "model": 77, "tool": [(15, "ok")]},
    "tool_down": {"queue": 8, "model": 30, "tool": [(30, "timeout"), (30, "timeout")]},
}


def run(case, profile, budget):
    source = SCENARIOS[case]
    clock = 0
    spans = []

    def span(name, milliseconds, status="ok"):
        nonlocal clock
        actual = min(milliseconds, budget - clock)
        outcome = status if actual == milliseconds else "deadline"
        spans.append({"operation": name, "start_ms": clock, "end_ms": clock + actual, "status": outcome})
        clock += actual
        return outcome

    def result(status):
        return {"case": case, "profile": profile, "result": status, "end_ms": clock, "spans": spans}

    for phase in ("queue", "model"):
        if span(phase, source[phase]) == "deadline":
            return result("deadline")

    for index, (tool_time, status) in enumerate(source["tool"], 1):
        if index == 2 and (profile == "baseline" or span("backoff", 10) == "deadline"):
            return result("timeout" if profile == "baseline" else "deadline")
        outcome = span(f"tool_attempt_{index}", tool_time, status)
        if outcome != "timeout":
            return result(outcome)
    return result("timeout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=SCENARIOS)
    parser.add_argument("profile", choices=["baseline", "one_retry"])
    parser.add_argument("--budget", type=int, default=110)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    print(json.dumps(run(args.case, args.profile, args.budget), indent=2))
