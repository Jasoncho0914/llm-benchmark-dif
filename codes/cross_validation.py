"""Training-question-only MH, held-out assignments, and ordinary question bootstrap.

No saved predictions or earlier result files are read. Every held-out question
keeps its own fold's model matching. Models, strata and fitted rules remain
fixed during question resampling. There is no random-subset comparator.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from .common import dif,matching
from . import plots

SPLIT_SEED=20261105
BOOTSTRAP_SEED=20261109
N_BOOT=1000


def model_contrast(labels,focal):
    """Coefficients yielding the pooled-count-weighted within-stratum gap."""
    result=np.zeros(len(focal))
    levels=np.unique(labels[labels<matching.UNMATCHED])
    masses=np.array([(labels==level).sum() for level in levels],float)
    masses/=masses.sum()
    for level,mass in zip(levels,masses):
        for family,sign in [(True,1),(False,-1)]:
            take=(labels==level)&(focal==family)
            result[take]=100*sign*mass/take.sum()
    return result


def feature_frame(study,benchmark,items):
    units=study.units.loc[items.unit_id]
    task=units.subtask.str.removeprefix(benchmark+'_').to_numpy()
    position=[f'{int(n)}:{int(p)}' for n,p in zip(units.n_options,units.correct_position)]
    return pd.DataFrame({'subcategory':task,'position':position})


def run(study):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    summary=[];curves=[];assignments=[];fold_results=[]
    for benchmark in ['bbh','musr']:
        _,items,_,_=study.primary(benchmark)
        x=study.responses(benchmark).astype(float)
        features=feature_frame(study,benchmark,items)
        n=x.shape[1];prediction=np.empty(n,object)
        fold_number=np.zeros(n,int);gap=np.zeros(n);folds=[]
        splitter=StratifiedKFold(5,shuffle=True,random_state=SPLIT_SEED)
        for fold,(train,test) in enumerate(splitter.split(features,features.subcategory),1):
            scores=x[:,train].mean(axis=1)
            labels,ranges=matching.strata(scores,study.focal)
            flags=dif.mh_common(x[:,train],labels,study.focal,study.codes,study.multipliers)
            large=flags.category.eq('C').to_numpy()&flags.delta.notna().to_numpy()
            target=np.where(flags.delta.to_numpy()[large]>0,'Qwen','Llama')
            assert len(np.unique(target))==2
            classifier=make_pipeline(OneHotEncoder(handle_unknown='ignore'),
                LogisticRegression(C=1,max_iter=2000,random_state=SPLIT_SEED))
            classifier.fit(features.iloc[train[large]],target)
            prediction[test]=classifier.predict(features.iloc[test]);fold_number[test]=fold
            gap[test]=model_contrast(labels,study.focal)@x[:,test]
            folds.append((fold,test,labels,ranges))
            fold_results.append(dict(benchmark=benchmark,fold=fold,training_questions=len(train),training_C=int(large.sum()),
                heldout_questions=len(test),predicted_Qwen=int((prediction[test]=='Qwen').sum()),predicted_Llama=int((prediction[test]=='Llama').sum())))
            print(f'{benchmark}: fold {fold}/5 complete',flush=True)
        assert (fold_number>0).all() and set(prediction)=={'Llama','Qwen'}
        assignment=items[['unit_id','source_position']].copy()
        assignment['benchmark']=benchmark;assignment['fold']=fold_number
        assignment['subcategory']=features.subcategory;assignment['position']=features.position
        assignment['predicted_favor']=prediction;assignment['matched_item_gap_pp']=gap
        assignments.append(assignment)
        # Identical sampling order to the ordinary-bootstrap figures in the paper.
        weights=rng.multinomial(n,np.full(n,1/n),size=N_BOOT)
        for pool,mask in [('All questions',np.ones(n,bool)),('Favor Qwen',prediction=='Qwen'),('Favor Llama',prediction=='Llama')]:
            selected=weights[:,mask];totals=selected.sum(axis=1)
            assert (totals>0).all()
            boot=(selected@gap[mask])/totals
            low,high=np.quantile(boot,[.025,.975])
            summary.append(dict(benchmark=benchmark,pool=pool,questions=int(mask.sum()),gap_pp=gap[mask].mean(),ci_low=low,ci_high=high))
        del weights
        for fold,test,labels,ranges in folds:
            size=len(test);weights=rng.multinomial(size,np.full(size,1/size),size=N_BOOT)
            heldout=x[:,test]
            for stratum,(low_score,high_score) in enumerate(ranges):
                q=(labels==stratum)&study.focal;l=(labels==stratum)&~study.focal
                item_gap=100*(heldout[q].mean(axis=0)-heldout[l].mean(axis=0))
                for pool,mask in [('All questions',np.ones(size,bool)),('Favor Qwen',prediction[test]=='Qwen'),('Favor Llama',prediction[test]=='Llama')]:
                    w=weights[:,mask];assert (w.sum(axis=1)>0).all()
                    boot=(w@item_gap[mask])/w.sum(axis=1)
                    low,high=np.quantile(boot,[.025,.975])
                    curves.append(dict(benchmark=benchmark,fold=fold,stratum=stratum,midpoint=(low_score+high_score)/2,
                        pool=pool,gap_pp=item_gap[mask].mean(),ci_low=low,ci_high=high))
            del weights
    summary=pd.DataFrame(summary);curves=pd.DataFrame(curves)
    study.results.update(crossfit_summary=summary,crossfit_curves=curves,
        crossfit_assignments=pd.concat(assignments,ignore_index=True),crossfit_folds=pd.DataFrame(fold_results))
    plots.overall_gaps(summary)
    plots.fold_gap_curves(curves)
    return summary
