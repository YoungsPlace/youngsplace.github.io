"""Compare full-prefix and KV-cached causal attention on a tiny CPU example."""

import math
from dataclasses import dataclass


# Four synthetic tokens, two features, one attention head. Each matrix row
# produces one output feature. These weights have not been trained.
EMBED = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
QUERY = ((0.0, 1.0), (-1.0, 0.0))
KEY = ((1.0, 0.5), (0.0, 1.0))
VALUE = ((0.5, 1.0), (-1.0, 0.5))
OUTPUT = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def project(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def softmax(scores):
    largest = max(scores)
    exp_scores = [math.exp(s - largest) for s in scores]
    total = sum(exp_scores)
    return [value / total for value in exp_scores]


@dataclass
class Work:
    kv_rows: int = 0
    qk_pairs: int = 0


def append_token(token, keys, values, work):
    position = len(keys)
    emb = EMBED[token]
    x = (emb[0] + 0.1 * position, emb[1] - 0.05 * position)
    q = project(QUERY, x)
    keys.append(project(KEY, x))
    values.append(project(VALUE, x))
    work.kv_rows += 1
    work.qk_pairs += len(keys)
    weights = softmax([dot(q, key) / math.sqrt(len(q)) for key in keys])
    context = tuple(sum(w * v[d] for w, v in zip(weights, values))
                    for d in range(len(q)))
    return project(OUTPUT, context)


def generate(prompt, new_tokens):
    if not prompt or new_tokens < 1:
        raise ValueError("Use a non-empty prompt and at least one output token.")
    prefix = list(prompt)
    keys, values = [], []
    full, cached = Work(), Work()
    for token in prefix:
        cache_logits = append_token(token, keys, values, cached)

    max_difference = 0.0
    for step in range(new_tokens):
        ref_keys, ref_values = [], []
        for token in prefix:
            ref_logits = append_token(token, ref_keys, ref_values, full)
        difference = max(abs(a - b) for a, b in zip(cache_logits, ref_logits))
        max_difference = max(max_difference, difference)
        assert difference < 1e-12
        chosen = max(range(len(cache_logits)), key=cache_logits.__getitem__)
        print(f"prefix={len(prefix)} cache_slots={len(keys)} next_id={chosen}")
        prefix.append(chosen)
        if step + 1 < new_tokens:
            cache_logits = append_token(chosen, keys, values, cached)
    print(f"full   kv_rows={full.kv_rows} qk_pairs={full.qk_pairs}")
    print(f"cached kv_rows={cached.kv_rows} qk_pairs={cached.qk_pairs}")
    print(f"max_abs_logit_diff={max_difference:.2e}")
    return prefix[len(prompt):]


if __name__ == "__main__":
    print("generated", generate([1, 3, 0], 3))
