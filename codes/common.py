"""Shared data and in-memory primary analysis; no intermediate files."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from . import data, matching, dif
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'output'
NAMES={'bbh':'BBH','gpqa':'GPQA','ifeval':'IFEval','openllm_math':'MATH','mmlu_pro':'MMLU-Pro','musr':'MuSR'}
def analyze_benchmark(name, matrix, manifest, models, cluster_codes, multipliers):
    """Screen questions, calculate matching scores, then estimate and flag DIF."""
    pooled_accuracy = matrix.mean(axis=0)
    keep = (pooled_accuracy >= 0.01) & (pooled_accuracy <= 0.99)
    responses = matrix[:, keep]
    questions = manifest.loc[keep].reset_index(drop=True)
    scores = responses.mean(axis=1)
    focal = models.group.eq('Qwen').to_numpy()
    strata, ranges = matching.strata(scores, focal)

    estimates = dif.mh_common(responses, strata, focal, cluster_codes, multipliers)
    items = pd.concat([questions, estimates], axis=1)
    items['pooled_accuracy'] = pooled_accuracy[keep]
    items['estimable'] = items.delta.notna()
    large = items[items.category.eq('C')]
    summary = {
        'benchmark': NAMES[name], 'items': len(items),
        'A_pct': 100 * items.category.eq('A').mean(),
        'B_pct': 100 * items.category.eq('B').mean(),
        'C_pct': 100 * items.category.eq('C').mean(),
        'C_favor_Llama_pct': 100 * large.direction.eq('Llama').mean(),
        'C_favor_Qwen_pct': 100 * large.direction.eq('Qwen').mean(),
        'median_abs_delta_C': large.delta.abs().median(),
        'nonestimable': int(items.delta.isna().sum()),
    }
    model_scores = pd.DataFrame({
        'benchmark': name, 'model': models.fullname, 'family': models.group,
        'matching_score': scores, 'stratum': strata,
        'matched': strata < matching.UNMATCHED,
    })
    boundaries = pd.DataFrame([
        {'benchmark': name, 'stratum': k, 'low_pct': low,
         'high_pct_exclusive': high,
         'llama_models': int(((strata == k) & ~focal).sum()),
         'qwen_models': int(((strata == k) & focal).sum())}
        for k, (low, high) in enumerate(ranges)
    ])
    return summary, items, model_scores, boundaries

class Study:
    """One loaded cohort, with primary item estimates reused by every analysis."""
    def __init__(self):
        self.models,self.manifests,self.matrices,self.lineage,_,_=data.load()
        self.units=pd.read_parquet(data.CURATED/'units.parquet').set_index('unit_id')
        self.codes,self.cluster_names=pd.factorize(self.lineage)
        self.multipliers=dif.bootstrap_multipliers(len(self.cluster_names),dif.SEED)
        self.focal=self.models.group.eq('Qwen').to_numpy()
        self.cache={}
        self.results={}
    def primary(self,benchmark):
        if benchmark not in self.cache:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=RuntimeWarning,message='.*(Degrees of freedom|All-NaN slice).*')
                self.cache[benchmark]=analyze_benchmark(benchmark,self.matrices[benchmark],self.manifests[benchmark],self.models,self.codes,self.multipliers)
        return self.cache[benchmark]
    def responses(self,benchmark):
        p=self.matrices[benchmark].mean(axis=0)
        return self.matrices[benchmark][:,(p>=.01)&(p<=.99)]
