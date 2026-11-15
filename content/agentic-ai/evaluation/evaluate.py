"""Score exact outputs, evidence, and modeled costs for both planning policies."""

import argparse
import json
from math import ceil, comb
from statistics import median

from replan import search

CASES = {
    "open": ("found", "A", "10:30"),
    "closed": ("found", "B", "14:00"),
    "empty": ("found", "B", "14:00"),
    "timeout_once": ("found", "A", "10:30"),
    "timeout_twice": ("timeout", None, None),
    "cycle": ("cycle", None, None),
    "no_slot": ("no_slot", None, None),
}
TOOL_MS = {"status": 10, "first_slot": 20}


def percentile95(values):
    return sorted(values)[ceil(0.95 * len(values)) - 1]


def evaluate(adaptive):
    results = []
    for case, gold in CASES.items():
        got = search(case, adaptive, max_calls=6)
        status, room, slot = got["status"], got["room"], got["slot"]
        log = got["trace"]
        output = (status, room if slot is not None else None, slot)
        evidence = slot is None or any(
            event["tool"] == "first_slot" and event["room"] == room and event["result"] == slot
            for event in log
        )
        ms = sum(60 if item["result"] == "timeout" else TOOL_MS[item["tool"]] for item in log)
        results.append({"case": case, "expected": gold, "output": output, "exact": output == gold,
                        "has_evidence": evidence, "calls": len(log), "sim_ms": ms})
    latencies = [result["sim_ms"] for result in results]
    return {"cases": results,
            "exact": sum(result["exact"] for result in results),
            "answer_count": sum(result["output"][2] is not None for result in results),
            "unsupported": sum(not result["has_evidence"] for result in results),
            "calls": sum(result["calls"] for result in results),
            "sim_p50_ms": median(latencies), "sim_p95_ms": percentile95(latencies)}


def paired_comparison(base, revised):
    pairs = list(zip(base["cases"], revised["cases"], strict=True))
    if any(a["case"] != b["case"] for a, b in pairs):
        raise ValueError("the reports must use the same cases in the same order")
    improved = sum(not a["exact"] and b["exact"] for a, b in pairs)
    regressed = sum(a["exact"] and not b["exact"] for a, b in pairs)
    discordant = improved + regressed
    tail = sum(comb(discordant, j) for j in range(min(improved, regressed) + 1))
    return {"improved": improved, "regressed": regressed, "tied": len(pairs) - discordant,
            "two_sided_exact_p": min(1, 2 * tail / 2 ** discordant)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    base, revised = evaluate(False), evaluate(True)
    if args.paired:
        print(json.dumps(paired_comparison(base, revised)))
    else:
        print(json.dumps({"mode": "fixed", **base}))
        print(json.dumps({"mode": "adaptive", **revised}))
