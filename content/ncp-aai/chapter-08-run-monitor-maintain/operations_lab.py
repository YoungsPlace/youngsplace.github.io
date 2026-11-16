"""Compare a release candidate with a baseline on paired synthetic traces."""

import json
from math import ceil
from statistics import median

# queue, model and tool milliseconds, and whether the task was correct
BASE = [(15, 160, 65, True), (30, 170, 70, True), (20, 150, 70, True),
        (45, 155, 75, True), (80, 170, 65, True), (25, 165, 75, True),
        (60, 170, 75, True), (70, 160, 75, True), (90, 175, 85, False), (160, 165, 85, True)]
CANDIDATE = [(15, 105, 65, True), (29, 106, 70, True), (20, 104, 70, True),
             (44, 102, 75, False), (67, 103, 65, True), (25, 105, 75, True),
             (50, 106, 75, True), (56, 102, 75, True), (78, 108, 85, False), (96, 108, 85, True)]


def metrics(trace):
    latencies = sorted(sum(row[:3]) for row in trace)
    return {"tasks_ok": sum(row[3] for row in trace), "p95_ms": latencies[ceil(.95 * len(trace)) - 1],
            "deadline_misses": sum(ms > 300 for ms in latencies),
            "median_queue_ms": median(row[0] for row in trace),
            "median_model_ms": median(row[1] for row in trace)}


if __name__ == "__main__":
    baseline, candidate = metrics(BASE), metrics(CANDIDATE)
    failures = [
        index for index, (a, b) in enumerate(zip(BASE, CANDIDATE, strict=True), 1)
        if a[3] and not b[3]
    ]
    ready = (candidate["tasks_ok"] >= baseline["tasks_ok"]
             and candidate["p95_ms"] <= 300 and not failures)
    print(json.dumps({"baseline": baseline, "candidate": candidate,
                      "regression_case_ids": failures, "decision": "promote" if ready else "hold"}))
