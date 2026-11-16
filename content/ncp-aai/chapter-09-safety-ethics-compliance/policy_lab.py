"""Exercise read/write authorization and a small synthetic fairness slice check."""

import json

ALLOWED_RECORDS = {"team-a": {"R42"}, "team-b": {"R41"}}


def authorize(action, team, record, revision, approved, audit):
    if action not in {"read", "publish"}:
        decision = "deny_action"
    elif record not in ALLOWED_RECORDS.get(team, set()):
        decision = "deny_scope"
    elif action == "publish" and (team, record, revision) != approved:
        decision = "hold_for_approval"
    else:
        decision = "allow"
    # Nothing here invokes a publish endpoint. Audit keeps structured IDs.
    audit.append({"action": action if action in {"read", "publish"} else "unknown",
                  "decision": decision, "record": record if record in ALLOWED_RECORDS.get(team, set()) else None})
    return decision


if __name__ == "__main__":
    audit = []
    matching = ("team-a", "R42", 3)
    policy_checks = [
        authorize("read", "team-a", "R42", 3, None, audit),
        authorize("read", "team-a", "R41", 3, None, audit),
        authorize("publish", "team-a", "R42", 4, matching, audit),
        authorize("publish", "team-a", "R42", 3, matching, audit),
    ]
    # Both groups contain benign requests. True means the filter blocked it.
    benign = {"direct_wording": (False, False, False, True),
              "indirect_wording": (True, True, False, False)}
    print(json.dumps({"checks": policy_checks, "audit": audit,
                      "benign_false_block_rates": {group: sum(rows) / len(rows) for group, rows in benign.items()}}))
