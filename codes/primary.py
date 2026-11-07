"""Primary MH tables, matching sensitivities and additional model families."""
import warnings
import numpy as np
import pandas as pd
from .common import data,dif,matching,NAMES
from . import tables


def prevalence(study):
    result=pd.DataFrame([study.primary(b)[0] for b in data.BENCHMARKS])
    rows=[]
    for r in result.itertuples(index=False):
        vals=[r.A_pct,r.B_pct,r.C_pct,r.C_favor_Llama_pct,r.C_favor_Qwen_pct]
        rows.append([r.benchmark,tables.count(r.items),*[tables.number(v) for v in vals],tables.number(r.median_abs_delta_C,2)])
    tables.write('DIF_prevalence.tex','tab:dif-summary',
        r'DIF prevalence and direction across benchmarks (primary analysis). A/B/C are classification shares. Directional shares and median absolute MH delta use Category-C items. Category A includes '+str(int(result.nonestimable.sum()))+' questions without finite estimates.',
        ['Benchmark','Items',r'\% A',r'\% B',r'\% C',r'\% C favor Llama',r'\% C favor Qwen',r'Median $|\Delta_{\mathrm{MH}}|$ (C)'],rows)
    study.results['prevalence']=result
    return result


def selection(study):
    raw=pd.read_parquet(data.CURATED/'models.parquet')
    selected=raw[raw.family.isin(['llama3_8b','qwen2_7b'])]
    logged=selected[selected.complete].sort_values('model')
    group=logged.identical_responses_group
    unique=logged[group.isna()|~group.duplicated(keep='first')]
    counts=[len(raw),len(selected),len(logged),len(unique),len(study.models)]
    steps=['Start with the curated dataset','Retain Llama-3-8B and Qwen2-7B families',
           'Require evaluation logs for all six benchmarks','Retain one model per identical correctness pattern',
           'Require responses to every retained question']
    rows=[[step,tables.count(n)] for step,n in zip(steps,counts)]
    for family in ['Llama','Qwen']:
        rows.append([family+' family in the final sample',tables.count(study.models.group.eq(family).sum())])
    tables.write('model_selection.tex','tab:model_selection','Model counts remaining after each selection step. Filters are applied sequentially. The final two rows give the family breakdown.',['Selection step','Model count'],rows,'lr')
    # The metadata also reproduce the configuration/uploader descriptions.
    metadata=raw.set_index('model').loc[study.models.fullname]
    verification=[]
    for family,part in study.models.groupby('group'):
        records=metadata.loc[part.fullname]
        available=records.config_hidden.notna() & records.config_layers.notna()
        uploaders=part.uploader.value_counts()
        verification.append(dict(family=family,models=len(part),configs=int(available.sum()),
            hidden_sizes=sorted(records.loc[available,'config_hidden'].unique().tolist()),
            layers=sorted(records.loc[available,'config_layers'].unique().tolist()),
            uploaders=len(uploaders),top_five_pct=100*uploaders.head(5).sum()/len(part),
            largest_uploader=uploaders.index[0],largest_uploader_models=int(uploaders.iloc[0]),
            largest_uploader_models_labeled_merge=int(part.loc[part.uploader.eq(uploaders.index[0]),'Type'].str.contains('merge',case=False,na=False).sum())))
    study.results['model_metadata']=pd.DataFrame(verification)
    base=study.models['Base Model'].astype('string').str.strip()
    missing=base.isna()|base.isin(['','?','Removed','nan'])
    study.results['lineage_summary']={'clusters':len(study.cluster_names),'missing_or_unspecified_singletons':int(missing.sum())}
    return pd.DataFrame({'step':steps,'models':counts})


def sensitivities(study):
    anchors=[];intervals=[]
    for b in data.BENCHMARKS:
        primary,items,scores,_=study.primary(b)
        y=study.responses(b);score=scores.matching_score.to_numpy()
        anchor_mask=items.category.eq('A').to_numpy()
        anchor_scores=y[:,anchor_mask].mean(axis=1)
        refined=dif.mh_common(y,matching.labels(anchor_scores,study.focal),study.focal,study.codes,study.multipliers)
        valid=items.delta.notna()&refined.delta.notna()
        large=refined[refined.category.eq('C')]
        anchors.append(dict(benchmark=NAMES[b],anchor_items=int(anchor_mask.sum()),
            primary_C_pct=primary['C_pct'],anchor_C_pct=100*refined.category.eq('C').mean(),
            primary_C_favor_qwen_pct=primary['C_favor_Qwen_pct'],
            anchor_C_favor_qwen_pct=100*large.direction.eq('Qwen').mean(),
            delta_correlation=np.corrcoef(items.delta[valid],refined.delta[valid])[0,1],
            same_category_pct=100*items.category.eq(refined.category).mean(),
            llama_matched_pct=100*scores.loc[scores.family.eq('Llama'),'matched'].mean(),
            qwen_matched_pct=100*scores.loc[scores.family.eq('Qwen'),'matched'].mean()))
        for width in [1,2,3,5]:
            labels=np.minimum(np.floor(score*100/width+1e-12).astype(int),99//width)
            size=int(labels.max())+1
            nr=np.bincount(labels[~study.focal],minlength=size)
            nf=np.bincount(labels[study.focal],minlength=size)
            supported=((nr>=5)&(nf>=5))[labels]
            result=dif.mh_common(y,labels,study.focal,study.codes,study.multipliers)
            c=result[result.category.eq('C')]
            intervals.append(dict(benchmark=NAMES[b],width=width,
                llama_matched_pct=100*supported[~study.focal].mean(),qwen_matched_pct=100*supported[study.focal].mean(),
                C_pct=100*result.category.eq('C').mean(),C_favor_qwen_pct=100*c.direction.eq('Qwen').mean()))
        print(NAMES[b]+': anchor and fixed-width sensitivity complete',flush=True)
    anchor=pd.DataFrame(anchors);interval=pd.DataFrame(intervals)
    rows=[]
    for b,g in interval.groupby('benchmark',sort=False):
        rows.append([b]+[f'{r.llama_matched_pct:.0f}/{r.qwen_matched_pct:.0f}' for r in g.itertuples()]+
                    [f'{r.C_pct:.0f}/{r.C_favor_qwen_pct:.0f}' for r in g.itertuples()])
    tables.write('DIF_interval_sensitivity.tex','tab:strata',
      r'Sensitivity to fixed score-interval width. Coverage columns give the percentage of models matched as Llama/Qwen pairs. The remaining columns give Category-C prevalence / the share of Category-C questions favoring Qwen. Primary matching uses merged intervals.',
      ['Benchmark']+[f'Matched: {w} pt' for w in [1,2,3,5]]+[f'C / Qwen: {w} pt' for w in [1,2,3,5]],rows)
    rows=[[r.benchmark,tables.count(r.anchor_items),tables.number(r.primary_C_pct),tables.number(r.anchor_C_pct),tables.number(r.delta_correlation,3),tables.number(r.same_category_pct)] for r in anchor.itertuples()]
    tables.write('DIF_anchor_sensitivity.tex','tab:anchor-sensitivity',
      r'Sensitivity to matching scores calculated from initial Category-A items. Correlation uses finite MH delta estimates; the last column gives the percentage retaining its A/B/C classification.',
      ['Benchmark','Anchors',r'Primary C (\%)',r'Anchor C (\%)',r'$r(\Delta)$',r'Same category (\%)'],rows)
    rows=[[r.benchmark,*[tables.number(v) for v in [r.llama_matched_pct,r.qwen_matched_pct,r.primary_C_pct,r.anchor_C_pct,r.primary_C_favor_qwen_pct,r.anchor_C_favor_qwen_pct]]] for r in anchor.itertuples()]
    tables.write('DIF_robustness_summary.tex','tab:rq1-robustness',
      r'Matching coverage and sensitivity to Category-A anchor scores. Coverage refers to primary merged intervals. Directional percentages use Category-C questions.',
      ['Benchmark',r'Llama matched (\%)',r'Qwen matched (\%)',r'Primary C (\%)',r'Anchor C (\%)',r'Primary C favor Qwen (\%)',r'Anchor C favor Qwen (\%)'],rows)
    study.results.update(anchors=anchor,intervals=interval)
    return anchor,interval


def additional_families(study):
    from .common import analyze_benchmark
    results=[]
    for focal,key in [('Qwen','qwen2_7b'),('Llama','llama3_8b')]:
        models,manifests,matrices,lineage,_,_=data.load(families=(key,'mistral_7b'))
        codes,names=pd.factorize(lineage);multipliers=dif.bootstrap_multipliers(len(names),dif.SEED)
        flag=models.group.eq(focal).to_numpy();rows=[]
        for b in data.BENCHMARKS:
            x=matrices[b];p=x.mean(axis=0);y=x[:,(p>=.01)&(p<=.99)]
            labels,_=matching.strata(y.mean(axis=1),flag)
            estimates=dif.mh_common(y,labels,flag,codes,multipliers)
            c=estimates[estimates.category.eq('C')]
            row=dict(comparison=focal+'--Mistral',benchmark=NAMES[b],items=y.shape[1],
                A_pct=100*estimates.category.eq('A').mean(),B_pct=100*estimates.category.eq('B').mean(),C_pct=100*estimates.category.eq('C').mean(),
                C_favor_reference_pct=100*c.direction.eq('Llama').mean(),C_favor_focal_pct=100*c.direction.eq('Qwen').mean(),
                median_abs_delta_C=c.delta.abs().median(),nonestimable=int(estimates.delta.isna().sum()),
                focal_matched_pct=100*(labels[flag]<matching.UNMATCHED).mean(),reference_matched_pct=100*(labels[~flag]<matching.UNMATCHED).mean())
            results.append(row)
            rows.append([NAMES[b],tables.count(y.shape[1]),*[tables.number(row[k]) for k in ['A_pct','B_pct','C_pct','C_favor_reference_pct','C_favor_focal_pct']],tables.number(row['median_abs_delta_C'],2)])
        tables.write(f'DIF_{focal}_Mistral.tex',f'tab:dif-{focal.lower()}-mistral',
          f'DIF prevalence and direction for {focal} ({int(flag.sum())} models) and Mistral ({int((~flag).sum())} models). Filters, matching and inference follow the primary procedure and are applied separately to each pair. Directional shares and effect sizes use Category-C items.',
          ['Benchmark','Items',r'\% A',r'\% B',r'\% C',r'\% C favor Mistral',r'\% C favor '+focal,r'Median $|\Delta_{\mathrm{MH}}|$ (C)'],rows)
        print(focal+'--Mistral complete',flush=True)
    study.results['additional_families']=pd.DataFrame(results)
    return study.results['additional_families']


def nonuniform(study):
    from .logistic_dif import fit_many_logistic
    results=[];summaries=[];dependence=[]
    for b in data.BENCHMARKS:
        _,items,scores,_=study.primary(b);y=study.responses(b)
        score=scores.matching_score.to_numpy();inside=scores.matched.to_numpy()
        p=np.full(y.shape[1],np.nan);low=p.copy();high=p.copy()
        beta=np.full((4,y.shape[1]),np.nan)
        for start in range(0,y.shape[1],1500):
            sl=slice(start,start+1500)
            coef,_,pp,ll,hh=fit_many_logistic(y[inside,sl].astype(float),score[inside],study.focal[inside],study.codes[inside])
            p[sl]=pp;low[sl]=ll;high[sl]=hh;beta[:,sl]=coef
        q=dif.bh(p);flag=q<=.05
        result=items[['unit_id','category']].copy()
        result['interaction_p']=p;result['interaction_q']=q;result['testable']=np.isfinite(p)
        result['gap_q25']=low;result['gap_q75']=high
        results.append(result.assign(benchmark=b))
        row=dict(benchmark=NAMES[b],items=len(items),tested=int(np.isfinite(p).sum()),nonuniform_pct_all=100*flag.mean())
        for cat in ['A','B','C']:
            take=items.category.eq(cat).to_numpy();row['nonuniform_pct_MH_'+cat]=100*flag[take].mean()
        bc=items.category.isin(['B','C']).to_numpy()
        dependence.append(dict(benchmark=NAMES[b],interaction_q05_pct=100*flag[bc].mean(),sign_change_pct=100*(low[bc]*high[bc]<0).mean()))
        summaries.append(row)
        print(NAMES[b]+': nonuniform DIF complete',flush=True)
    result=pd.DataFrame(summaries)
    rows=[[r.benchmark,tables.count(r.items),tables.count(r.tested),*[tables.number(v) for v in [r.nonuniform_pct_all,r.nonuniform_pct_MH_A,r.nonuniform_pct_MH_B,r.nonuniform_pct_MH_C]]] for r in result.itertuples()]
    tables.write('DIF_nonuniform.tex','tab:nonuniform-dif',
      r'Nonuniform DIF between Llama and Qwen. Flags require a BH-adjusted family-by-score interaction $p\leq0.05$. Percentages use all questions or all questions in the specified primary MH category. Untestable fits remain in these denominators but are not flagged.',
      ['Benchmark','Items','Tested',r'Nonuniform (\%)',r'Within MH A (\%)',r'Within MH B (\%)',r'Within MH C (\%)'],rows)
    study.results.update(nonuniform=result,score_dependence=pd.DataFrame(dependence),logistic_items=pd.concat(results,ignore_index=True))
    return result
