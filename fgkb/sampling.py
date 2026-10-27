"""Sampling helpers (Quasi-Monte Carlo)."""

import numpy as np

def halton_sequence(b):
    """ref: https://en.wikipedia.org/wiki/Halton_sequence"""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

def gen_quasi_mc_points(d, N, lows, highs):
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  ## Available for dimensions up to 10. For higher dimensions, more prime bases can be added.
    if d > len(bases):
        raise ValueError(f"Dimension {d} is too high for the available bases.")
    halton_generators = [halton_sequence(bases[i]) for i in range(d)]

    points = []
    for _ in range(N):
        point = [next(halton_generators[i]) * (highs[i] - lows[i]) + lows[i] for i in range(d)]
        points.append(point)
    return np.array(points)

