"""Notebook entry points and a strict list of the paper's final outputs."""
from pathlib import Path
import platform,time,resource
import numpy as np
import pandas as pd
from .common import ROOT,OUTPUT,Study
from . import primary,content,cross_validation
from .examples import item_examples

PAPER_OUTPUTS={
 'DIF_prevalence.tex','DIF_category_concentration.tex','DIF_item_examples.tex',
 'model_selection.tex','DIF_interval_sensitivity.tex','DIF_anchor_sensitivity.tex',
 'DIF_robustness_summary.tex','DIF_Qwen_Mistral.tex','DIF_Llama_Mistral.tex',
 'DIF_nonuniform.tex','DIF_position_logistic.tex','DIF_BBH_answer_selection.tex',
 'DIF_BBH_task_answer.tex','fig_bbh_answer_selection.pdf','overall_rq3.png',
 'bbh_gap_curves_large_text.pdf','musr_gap_curves_large_text.pdf',
 'fig_category_dif_bbh.pdf','fig_category_dif_mmlupro.pdf',
}


def start():
    """Clear generated outputs only, then load the provided raw input tables."""
    OUTPUT.mkdir(exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file() or path.is_symlink():path.unlink()
        else:
            raise RuntimeError(f'Unexpected directory in output: {path}. Move it before running.')
    started=time.perf_counter()
    study=Study();study.started=started
    print(f'Loaded {len(study.models)} models and {len(study.cluster_names)} lineage clusters.')
    print('Python',platform.python_version(),'on',platform.platform())
    return study


def validate(study):
    """Check paper-level results and ensure no intermediate exports were saved."""
    assert study.models.group.value_counts().to_dict()=={'Llama':909,'Qwen':371}
    assert study.results['prevalence']['items'].sum()==19563
    assert np.allclose(study.results['prevalence'].C_pct.round(1),[57.1,51.3,35.7,7.2,40.8,63.8])
    assert np.allclose(study.results['nonuniform'].nonuniform_pct_all.round(1),[66.9,60.7,45.2,28.8,60.5,45.4])
    cv=study.results['crossfit_summary'].set_index(['benchmark','pool'])
    for benchmark,pool,value in [('bbh','All questions',.14),('bbh','Favor Qwen',11.27),
        ('bbh','Favor Llama',-10.65),('musr','All questions',.46),('musr','Favor Qwen',6.46),('musr','Favor Llama',-9.36)]:
        assert np.isclose(cv.loc[(benchmark,pool),'gap_pp'],value,atol=.0051)
    got={p.name for p in OUTPUT.iterdir() if p.is_file()}
    assert got==PAPER_OUTPUTS,{'missing':PAPER_OUTPUTS-got,'unexpected':got-PAPER_OUTPUTS}
    elapsed=time.perf_counter()-study.started
    # macOS reports bytes; Linux reports KiB.
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib=peak/(1024**3 if platform.system()=='Darwin' else 1024**2)
    print(f'Validated {len(got)} paper artifacts. Wall time: {elapsed/60:.2f} minutes; peak process memory: {gib:.2f} GiB.')
    print('All intermediate estimates and predictions remain in memory; no CSV, cache or duplicate image exports were written.')
    return pd.DataFrame({'paper_output':sorted(got)})
