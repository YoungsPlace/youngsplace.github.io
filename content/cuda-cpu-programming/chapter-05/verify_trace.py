"""Check the incremental demo against full-matrix masked attention with NumPy."""

from itertools import product

import numpy as np

from decode_trace import EMBED, KEY, OUTPUT, QUERY, VALUE, Work, append_token


def all_position_logits(tokens):
    length = len(tokens)
    positions = np.arange(length)[:, None]
    x = np.asarray(EMBED)[list(tokens)] + positions * np.array([0.1, -0.05])
    q = x @ np.transpose(QUERY)
    k = x @ np.transpose(KEY)
    v = x @ np.transpose(VALUE)
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores[np.triu_indices(length, 1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v @ np.transpose(OUTPUT)


def verify():
    cases = 0
    largest_error = 0.0
    for length in range(1, 5):
        for tokens in product(range(len(EMBED)), repeat=length):
            expected = all_position_logits(tokens)
            keys, values = [], []
            work = Work()
            actual = np.array([append_token(token, keys, values, work) for token in tokens])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            largest_error = max(largest_error, float(np.max(np.abs(actual - expected))))
            cases += 1
    print(f"prefixes={cases} max_abs_logit_diff={largest_error:.3e}")


if __name__ == "__main__":
    verify()
