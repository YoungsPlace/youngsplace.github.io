"""Compare sequential tool calls and a fan-out on synthetic room reports."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, sleep
from urllib.request import Request, urlopen

DELAYS = {"hours": 0.04, "access": 0.07, "equipment": 0.09}
RESULTS = {
    "hours": {"value": "09:00–18:00", "source": "hours-v3"},
    "access": {"value": "campus card", "source": "access-v2"},
    "equipment": {"value": "soldering station", "source": "equipment-v1"},
}
REQUESTS = {"hours": ("hours",), "visit": ("hours", "access", "equipment")}
QUERIES = {
    "hours": "When is the lab open?",
    "visit": "Give me the opening hours, entry requirements, and available lab equipment.",
    "equipment_and_access": "I already know the opening hours. What equipment is available, and which ID should I bring?",
}


def read(name, fault):
    sleep(DELAYS[name])
    if fault and name == "equipment":
        return {"status": "timeout"}
    return {"status": "ok", **RESULTS[name]}


def choose(query):
    schema = {
        "type": "object",
        "properties": {"tools": {"type": "array", "items": {"enum": list(RESULTS)},
                                "minItems": 1, "maxItems": len(RESULTS)}},
        "required": ["tools"],
        "additionalProperties": False,
    }
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {"role": "system", "content": "Choose only lab tools whose facts the user asks for. hours = opening times; access = entry conditions or identity card; equipment = lab tools. Return JSON with the tools array."},
            {"role": "user", "content": query},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 80},
    }
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        names = json.loads(json.load(response)["message"]["content"])["tools"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or name not in RESULTS for name in names):
        raise ValueError("expected allowed read tools")
    if len(names) != len(set(names)):
        raise ValueError("duplicate request for a read tool")
    return names


def report(request, mode, fault, max_workers=3):
    names = choose(QUERIES[request]) if mode == "dispatch" else REQUESTS[request]
    start = perf_counter()
    if mode in {"parallel", "dispatch"}:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {name: pool.submit(read, name, fault) for name in names}
            parts = {name: future.result() for name, future in futures.items()}
    else:
        parts = {name: read(name, fault) for name in names}
    return {"status": "complete" if all(part["status"] == "ok" for part in parts.values()) else "partial",
            "calls": len(parts), "parts": parts, "ms": round((perf_counter() - start) * 1000, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", choices=QUERIES)
    parser.add_argument("mode", choices=["serial", "parallel", "dispatch"])
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.request not in REQUESTS and args.mode != "dispatch":
        parser.error("this free-text question requires dispatch mode")
    print(json.dumps(report(args.request, args.mode, args.fault, args.max_workers)))
