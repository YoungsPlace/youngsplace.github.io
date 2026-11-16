"""Replay UI events for one synthetic run without performing an external write."""

from dataclasses import asdict, dataclass
import json


@dataclass
class Run:
    phase: str = "needs_input"
    revision: int = 0
    baseline: str | None = None
    approval: int | None = None


def apply(run, event, version, value):
    if event == "choose_baseline" and run.phase == "needs_input" and value in {"R38", "R41"}:
        run.phase, run.baseline = "in_review", value
        run.revision += 1
        return "applied"
    if event == "edit" and run.phase in {"in_review", "approved"} and version == run.revision:
        run.revision += 1
        run.phase, run.approval = "in_review", None
        return "applied"
    if event == "approve" and run.phase == "in_review" and version == run.revision:
        run.phase, run.approval = "approved", version
        return "applied"
    if event == "approve" and run.phase == "approved" and version == run.approval:
        return "already_applied"
    if event == "cancel" and run.phase in {"in_review", "approved"}:
        run.phase, run.approval = "cancelled", None
        return "applied"
    return "ignored_stale_or_invalid"


if __name__ == "__main__":
    run = Run()
    events = (
        ("choose_baseline", 0, "R38"), ("approve", 1, None), ("approve", 1, None),
        ("edit", 1, None), ("approve", 1, None), ("approve", 2, None),
        ("cancel", 2, None), ("approve", 2, None),
    )
    for count, (event, version, value) in enumerate(events, 1):
        status = apply(run, event, version, value)
        print(json.dumps({"sequence": count, "event": event, "status": status, **asdict(run)}))
