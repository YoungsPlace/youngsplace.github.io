"""Compare queue wait and admission on a deterministic virtual timeline."""

import heapq
import json
from math import ceil

ARRIVALS = (0, 0, 10, 15, 25, 80, 120)
SERVICE = (70, 55, 60, 95, 40, 65, 50)


def simulate(replicas, max_wait, deadline=150):
    ready = [(0, worker) for worker in range(replicas)]
    heapq.heapify(ready)
    rows = []
    for req, (arrival, service) in enumerate(zip(ARRIVALS, SERVICE, strict=True), 1):
        free, worker = ready[0]
        start = max(free, arrival)
        wait = start - arrival
        if max_wait is not None and wait > max_wait:
            rows.append({"request": req, "status": "rejected", "projected_wait": wait})
            continue
        finish = start + service
        heapq.heapreplace(ready, (finish, worker))
        rows.append({"request": req, "worker": worker, "wait": wait, "duration": finish - arrival,
                     "status": "ok" if finish - arrival <= deadline else "deadline"})
    times = sorted(row["duration"] for row in rows if "duration" in row)
    return {"replicas": replicas, "max_wait": max_wait, "ok": sum(row["status"] == "ok" for row in rows),
            "deadline": sum(row["status"] == "deadline" for row in rows),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "admitted_p95_ms": times[ceil(.95 * len(times)) - 1], "trace": rows}


if __name__ == "__main__":
    for config in [(1, None), (2, None), (2, 40)]:
        print(json.dumps(simulate(*config)))
