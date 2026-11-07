"""Primary MH estimator, lineage bootstrap and category rules.
Numerical functions are retained verbatim from the coauthor's estimator.
Positive delta favors Qwen; category A preserves the missing-effect fallback.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
MIN_PER_GROUP=5
N_BOOT=1000
SEED=20261030
DELTA_SCALE=2.35

def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values, preserving NaNs."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    good = np.isfinite(p)
    values = p[good]
    if not len(values):
        return out
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1)
    out[good] = restored
    return out

def bootstrap_multipliers(n_cluster: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1 / n_cluster, n_cluster)
    return rng.multinomial(n_cluster, probabilities, size=N_BOOT).astype(np.float32) - 1

def mh_common(
    y: np.ndarray,
    labels: np.ndarray,
    focal: np.ndarray,
    clusters: np.ndarray,
    multipliers: np.ndarray,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Vectorized MH estimates and cluster-linearized bootstrap inference."""
    n, j = y.shape
    weights = np.ones(n) if weights is None else np.asarray(weights, float)
    k = int(labels.max()) + 1
    raw_cell = labels + k * focal.astype(int)
    raw_membership = sparse.csr_matrix(
        (np.ones(n), (raw_cell, np.arange(n))), shape=(2 * k, n)
    )
    raw_total = np.asarray(raw_membership.sum(axis=1)).reshape(2, k)
    use = (raw_total[0] >= MIN_PER_GROUP) & (raw_total[1] >= MIN_PER_GROUP)

    membership = sparse.csr_matrix(
        (weights, (raw_cell, np.arange(n))), shape=(2 * k, n)
    )
    correct = np.asarray(membership @ y).reshape(2, k, j)
    total = np.asarray(membership.sum(axis=1)).reshape(2, k)
    a, c = correct[0], correct[1]
    b = total[0, :, None] - a
    d = total[1, :, None] - c
    nn = (total[0] + total[1])[:, None]

    numerator = (a[use] * d[use] / nn[use]).sum(axis=0)
    denominator = (b[use] * c[use] / nn[use]).sum(axis=0)
    estimable = (numerator > 0) & (denominator > 0)
    delta = np.full(j, np.nan)
    delta[estimable] = -DELTA_SCALE * np.log(
        numerator[estimable] / denominator[estimable]
    )

    p_ref = a[use] / total[0, use, None]
    p_focal = c[use] / total[1, use, None]
    shares = nn[use, 0] / nn[use, 0].sum()
    pdiff = shares @ (p_focal - p_ref)

    # Linearized influence of a cluster-weight perturbation on MH delta.
    n_cluster = multipliers.shape[1]
    influence = np.zeros((n_cluster, j), dtype=np.float32)
    for i in range(n):
        s = labels[i]
        if not use[s]:
            continue
        w = weights[i]
        response = y[i]
        if focal[i]:
            d_num = w * (a[s] * (1 - response) / nn[s] - a[s] * d[s] / nn[s] ** 2)
            d_den = w * (b[s] * response / nn[s] - b[s] * c[s] / nn[s] ** 2)
        else:
            d_num = w * (response * d[s] / nn[s] - a[s] * d[s] / nn[s] ** 2)
            d_den = w * ((1 - response) * c[s] / nn[s] - b[s] * c[s] / nn[s] ** 2)
        row = np.zeros(j, dtype=np.float32)
        row[estimable] = -DELTA_SCALE * (
            d_num[estimable] / numerator[estimable]
            - d_den[estimable] / denominator[estimable]
        )
        influence[clusters[i]] += row

    se = np.full(j, np.nan)
    lo = np.full(j, np.nan)
    hi = np.full(j, np.nan)
    batch = 512
    for start in range(0, j, batch):
        stop = min(start + batch, j)
        perturbation = multipliers @ influence[:, start:stop]
        draws = delta[start:stop][None, :] + perturbation
        se[start:stop] = np.nanstd(draws, axis=0, ddof=1)
        lo[start:stop] = np.nanquantile(draws, 0.025, axis=0)
        hi[start:stop] = np.nanquantile(draws, 0.975, axis=0)

    coverage = np.isin(labels, np.flatnonzero(use)).mean()
    return classify(
        pd.DataFrame(
            {
                "delta": delta,
                "conditional_gap": pdiff,
                "se_cluster": se,
                "ci_low": lo,
                "ci_high": hi,
                "strata_used": int(use.sum()),
                "model_coverage": coverage,
            }
        )
    )

def classify(result: pd.DataFrame) -> pd.DataFrame:
    d = result.delta.to_numpy()
    se = result.se_cluster.to_numpy()
    z0 = np.divide(np.abs(d), se, out=np.full_like(d, np.nan), where=se > 0)
    p0 = 2 * norm.sf(z0)
    z1 = np.divide(np.abs(d) - 1, se, out=np.full_like(d, np.nan), where=se > 0)
    p1 = np.where(np.abs(d) > 1, norm.sf(z1), 1.0)
    q0, q1 = bh(p0), bh(p1)
    category_c = (np.abs(d) >= 1.5) & (q1 <= 0.05)
    category_b = (~category_c) & (np.abs(d) >= 1) & (q0 <= 0.05)
    result["p_zero"] = p0
    result["q_zero"] = q0
    result["p_beyond_one"] = p1
    result["q_beyond_one"] = q1
    result["category"] = np.where(category_c, "C", np.where(category_b, "B", "A"))
    result["direction"] = np.where(d > 0, "Qwen", np.where(d < 0, "Llama", "None"))
    return result


def estimate(responses, focal, lineage, score=None):
    """Run MH with merged score intervals and recorded-lineage uncertainty.

    With score=None, use mean correctness on responses (the primary analysis).
    To perform one-step anchor sensitivity, pass the mean on preliminary A items.
    Question filtering is the caller's responsibility.
    """
    from .matching import labels
    codes, cluster_names = pd.factorize(lineage)
    multipliers = bootstrap_multipliers(len(cluster_names), SEED)
    if score is None:
        score = responses.mean(axis=1)
    return mh_common(responses, labels(score, focal), focal, codes, multipliers)
