"""Compare local streaming latency with Ollama's load, prefill and decode counters."""

import argparse
import json
from statistics import median
from time import perf_counter_ns
from urllib.request import Request, urlopen

PROMPT = "Write a numbered list of 40 science topics. Put each topic on its own line."
PREFIX = "This line is padding for a laboratory latency measurement.\n"


def measure(model, lines, identity=""):
    prompt = identity + PREFIX * lines + PROMPT
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "5m",
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 48},
    }
    request = Request("http://127.0.0.1:11434/api/generate",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = perf_counter_ns()
    first = None
    final = None
    with urlopen(request, timeout=45) as response:
        for line in response:
            chunk = json.loads(line)
            if first is None and chunk.get("response"):
                first = perf_counter_ns()
            if chunk.get("done"):
                final = chunk
                break
    elapsed = perf_counter_ns() - started
    if first is None or final is None:
        raise RuntimeError("expected streamed output and final counters")
    nanoseconds = {"ttft": first - started, "e2e": elapsed, "server": final["total_duration"],
                   "load": final["load_duration"], "prefill": final["prompt_eval_duration"],
                   "decode": final["eval_duration"]}
    return {"prompt_tokens": final["prompt_eval_count"], "output_tokens": final["eval_count"],
            **{f"{key}_ms": round(value / 1e6, 3) for key, value in nanoseconds.items()}}


def summarize(rows):
    return {key: median([row[key] for row in rows]) for key in rows[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--distinct-prefix", action="store_true")
    args = parser.parse_args()
    measure(args.model, 0)
    settings = {"short": 0, "long": 60}
    runs = {key: [] for key in settings}
    for trial in range(5):
        for position, (key, lines) in enumerate(settings.items()):
            identity = f"{10 + 2 * trial + position}. " if args.distinct_prefix else ""
            runs[key].append(measure(args.model, lines, identity))
    print(json.dumps({key: {"runs": rows, "median": summarize(rows)} for key, rows in runs.items()}, indent=2))
