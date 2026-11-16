"""Run a synthetic, read-only report workflow with local Ollama."""

import argparse
import base64
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

RUNS = {
    "R41": {"detection_pct": 92.0, "source": "synthetic:R41@1"},
    "R42": {"detection_pct": 84.0, "source": "synthetic:R42@1"},
}

INTENTS = ("lookup", "compare", "unsupported")
NULLABLE_ID = {"type": ["string", "null"]}
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "target": NULLABLE_ID,
        "baseline": NULLABLE_ID,
    },
    "required": ["intent", "target", "baseline"],
    "additionalProperties": False,
}
IMAGE_SCHEMA = {
    "type": "object",
    "properties": {"target": NULLABLE_ID, "baseline": NULLABLE_ID},
    "required": ["target", "baseline"],
    "additionalProperties": False,
}
COMMON = """Classify this request for a read-only test report workflow.
Use lookup for reading one run, compare for comparing two runs,
and unsupported for external changes or publishing.
An ID is written explicitly as R followed by digits.
Use null for missing IDs. For lookup, use null for baseline.
For unsupported, use null for both IDs.
"""
PROMPTS = {
    "v1": COMMON + "For a comparison, use the first mentioned ID as target and the second ID as baseline.",
    "v2": COMMON + """For a comparison:
- The target is the run being evaluated. The baseline is the reference.
- Explicit labels such as 'baseline', 'reference', and 'target' determine the roles,
  regardless of where they occur in the sentence.
- In 'compare X against Y', X is the target and Y is the baseline.
- If only one ID is provided, keep the other role null. Never invent a run.
""",
}
PROMPTS["v3"] = PROMPTS["v2"] + """
Decide the intent before assigning IDs. If the user only asks to show, get,
or read one result without a comparative request, choose lookup.
Examples:
- 'Read the recorded result for R17.' -> lookup, target R17, baseline null.
- 'Compare R17 with a baseline.' -> compare, target R17, baseline null.
"""

CASES = [
    ("lookup", "Show the result of R42.", ("lookup", "R42", None), "ok", 0),
    ("compare", "Compare R42 against R41.", ("compare", "R42", "R41"), "ok", 0),
    ("baseline_first", "Use R41 as the baseline and compare it with target R42.",
     ("compare", "R42", "R41"), "ok", 0),
    ("reverse_reference", "With reference R42, how did target R41 compare?",
     ("compare", "R41", "R42"), "ok", 0),
    ("missing_baseline", "Compare R42 with the baseline.", ("compare", "R42", None), "needs_input", 0),
    ("unknown_run", "Show the result of R99.", ("lookup", "R99", None), "not_found", 0),
    ("unsupported", "Publish the R42 report.", ("unsupported", None, None), "unsupported", 0),
    ("transient", "Read the result for R41.", ("lookup", "R41", None), "ok", 1),
    ("timeout", "Get the detection rate for R42.", ("lookup", "R42", None), "tool_unavailable", 2),
]


def chat(model, system, message, schema):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, message],
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 96},
    }
    request = Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=45) as response:
        return json.loads(json.load(response)["message"]["content"])


def valid_plan(proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"intent", "target", "baseline"}:
        return False
    return (proposal["intent"] in INTENTS
            and all(value is None or isinstance(value, str) for value in
                    (proposal["target"], proposal["baseline"])))


def execute(proposal, mock_timeouts=0):
    trace = []

    def result(status, **fields):
        return {"status": status, **fields, "trace": trace}

    if not valid_plan(proposal):
        return result("invalid_plan")
    intent, target, baseline = (proposal[key] for key in ("intent", "target", "baseline"))
    if intent == "unsupported":
        return result("unsupported")
    if any(value is not None and not re.fullmatch(r"R[0-9]+", value) for value in (target, baseline)):
        return result("invalid_plan")
    if not target or (intent == "compare" and not baseline):
        return result("needs_input")

    def read(run_id):
        nonlocal mock_timeouts
        for attempt in (1, 2):
            if mock_timeouts:
                mock_timeouts -= 1
                status = "timeout"
            else:
                status = "ok" if run_id in RUNS else "not_found"
            trace.append({"tool": "read_run", "run_id": run_id, "attempt": attempt, "status": status})
            if status != "timeout":
                return status, RUNS.get(run_id)
        return "tool_unavailable", None

    status, current = read(target)
    if status != "ok":
        return result(status)
    if intent == "lookup":
        return result("ok", target=target, detection_pct=current["detection_pct"],
                      evidence=[current["source"]])

    status, reference = read(baseline)
    if status != "ok":
        return result(status)
    return result("ok", target=target, baseline=baseline,
                  delta_pp=round(current["detection_pct"] - reference["detection_pct"], 3),
                  evidence=[current["source"], reference["source"]])


def assess(model, prompt_version):
    for key, question, gold, status, timeouts in CASES:
        try:
            proposal = chat(model, PROMPTS[prompt_version], {"role": "user", "content": question}, ROUTE_SCHEMA)
            output = execute(proposal, timeouts)
        except (OSError, ValueError, TypeError, KeyError) as error:
            proposal = None
            output = {"status": "model_error", "error_type": type(error).__name__, "trace": []}
        actual = tuple(proposal[k] for k in ("intent", "target", "baseline")) if valid_plan(proposal) else None
        print(json.dumps({"case": key, "prompt": prompt_version, "gold": gold, "proposal": proposal,
                          "plan_match": actual == gold, "status_match": output["status"] == status,
                          "output": output}))


def image_ids(model, path):
    message = {
        "role": "user",
        "content": "Extract the TARGET RUN and BASELINE RUN identifiers from the image.",
        "images": [base64.b64encode(path.read_bytes()).decode()],
    }
    system = ("Read only the provided image. Return target and baseline IDs. "
              "If a baseline run is absent or unreadable, set baseline to null. Never guess an ID.")
    ids = chat(model, system, message, IMAGE_SCHEMA)
    print(json.dumps(ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--prompt", choices=PROMPTS, default="v3")
    parser.add_argument("--image", type=Path, help="Extract IDs from a synthetic panel rather than running text cases")
    args = parser.parse_args()
    if args.image:
        image_ids(args.model, args.image)
    else:
        assess(args.model, args.prompt)
