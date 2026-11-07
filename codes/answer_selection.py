"""Shared OLS adjustment and ordinary lineage-cluster bootstrap for choices."""
import numpy as np
import pandas as pd

N_BOOT = 1000
SEED = 20261106

def adjusted_selection(per, lineage):
    """Resample whole lineage clusters and refit rate ~ 1 + Qwen + score.

    All members of a sampled cluster receive its sampling multiplicity. Both
    families stay in the same regression. Questions, rates and scores are fixed.
    Returns percentile intervals, not normal-approximation/HC3 intervals.
    """
    groups = lineage.reindex(per.index)
    assert groups.notna().all() and per.notna().all().all()
    codes, names = pd.factorize(groups)
    X = np.column_stack([np.ones(len(per)), per.qwen, per.score])
    y = per.rate.to_numpy(float)
    coefficient = np.linalg.lstsq(X, y, rcond=None)[0][1]
    # Cluster sufficient statistics make exact weighted refits inexpensive.
    xx = np.zeros((len(names),3,3)); xy = np.zeros((len(names),3))
    np.add.at(xx, codes, X[:,:,None]*X[:,None,:])
    np.add.at(xy, codes, X*y[:,None])
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(names), np.full(len(names),1/len(names)),size=N_BOOT)
    draws = np.linalg.solve(np.einsum('bc,cij->bij',counts,xx), counts@xy)[:,1]
    lo,hi = np.quantile(draws,[.025,.975])
    return dict(adjusted_difference_pp=100*coefficient,low_pp=100*lo,high_pp=100*hi,
                bootstrap_draws=N_BOOT,bootstrap_seed=SEED,lineage_clusters=len(names))
