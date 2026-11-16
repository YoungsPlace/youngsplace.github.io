"""Compare two abstention thresholds using fixed synthetic proposal scores."""

import json
from collections import Counter

# task, confidence, answer expected, proposal correct if accepted
DATA = {
    "development": [
        ("lookup", .92, True, True), ("compare", .62, True, True),
        ("compare", .83, True, False), ("lookup", .88, True, True),
        ("ambiguous", .78, False, False), ("no_evidence", .55, False, False),
    ],
    "holdout": [
        ("lookup", .94, True, True), ("compare", .76, True, True),
        ("compare", .87, True, True), ("lookup", .81, True, False),
        ("ambiguous", .72, False, False), ("ambiguous", .91, False, False),
        ("no_evidence", .46, False, False), ("no_evidence", .61, False, False),
    ],
}


def score(rows, threshold):
    by_task = {}
    total = Counter()
    for task, confidence, answer_expected, correct in rows:
        if confidence >= threshold:
            label = "correct_answer" if answer_expected and correct else (
                "wrong_answer" if answer_expected else "unsupported_answer")
        else:
            label = "missed_answer" if answer_expected else "correct_abstention"
        total[label] += 1
        by_task.setdefault(task, Counter())[label] += 1
    answered = total["correct_answer"] + total["wrong_answer"] + total["unsupported_answer"]
    return {"cases": len(rows), "answered": answered,
            "task_success": total["correct_answer"] + total["correct_abstention"],
            "selective_accuracy": round(total["correct_answer"] / answered, 4) if answered else None,
            "unsupported_answers": total["unsupported_answer"], "by_task": by_task}


if __name__ == "__main__":
    for name, rows in DATA.items():
        for threshold in (.60, .80):
            print(json.dumps({"dataset": name, "threshold": threshold, **score(rows, threshold)}))
