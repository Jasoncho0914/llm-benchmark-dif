"""Score strata for Mantel--Haenszel matching ("thick matching").

Models are grouped by their score in 1-percentage-point steps. Starting from the
lowest score, neighboring steps are merged until the interval contains at least
MINIMUM models from each group; an interval may not be wider than CAP points, and
when it would be, its lowest step is left unmatched and merging restarts at the
next step. Models in steps that never reach the minimum are unmatched: they get a
label of their own, so the MH estimator drops them with every stratum that lacks
MINIMUM models from each group.
"""

from __future__ import annotations

import numpy as np

MINIMUM = 5
CAP = 5
UNMATCHED = 1000


def strata(score: np.ndarray, focal: np.ndarray, minimum: int = MINIMUM, cap: int = CAP):
    """Stratum label per model and the (low, high) score range of each stratum, in points."""
    points = np.minimum(np.floor(np.asarray(score) * 100 + 1e-12).astype(int), 100)
    focal = np.asarray(focal, dtype=bool)
    n_ref = np.bincount(points[~focal], minlength=101)
    n_foc = np.bincount(points[focal], minlength=101)
    step_label = UNMATCHED + np.arange(101)
    ranges = []
    start = 0
    while start <= 100:
        end, a, b = start, 0, 0
        while end <= 100 and end - start < cap:
            a += n_ref[end]
            b += n_foc[end]
            if a >= minimum and b >= minimum:
                break
            end += 1
        if end <= 100 and end - start < cap and a >= minimum and b >= minimum:
            step_label[start:end + 1] = len(ranges)
            ranges.append((start, end + 1))
            start = end + 1
        else:
            start += 1
    return step_label[points], ranges


def labels(score: np.ndarray, focal: np.ndarray) -> np.ndarray:
    return strata(score, focal)[0]
