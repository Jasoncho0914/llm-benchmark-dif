"""Content-category tests, answer-position regressions and BBH actual choices."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, chi2
from .common import OUTPUT,NAMES,data,dif
from .position import firth_logistic
from .answer_selection import adjusted_selection
from . import tables,plots

def category_labels(benchmark, questions, units):
    """Assign exactly one content category per question."""
    if benchmark == 'bbh':
        return units.loc[questions.unit_id, 'subtask'].str.removeprefix('bbh_').to_numpy()
    if benchmark == 'gpqa':
        return units.loc[questions.unit_id, 'domain'].to_numpy()
    if benchmark == 'openllm_math':
        return units.loc[questions.unit_id, 'problem_type'].to_numpy()
    if benchmark == 'mmlu_pro':
        return units.loc[questions.unit_id, 'subject'].to_numpy()
    if benchmark == 'musr':
        return units.loc[questions.unit_id, 'subtask'].str.removeprefix('musr_').to_numpy()
    if benchmark == 'ifeval':
        instruction_lists = units.loc[questions.unit_id, 'instruction_ids']
        return [str(ids[0]).split(':')[0] if len(ids) else 'none' for ids in instruction_lists]
    raise ValueError(benchmark)

def test_table(table):
    if min(table.shape) < 2:
        return np.nan, np.nan, np.nan
    _, p, _, expected = chi2_contingency(table)
    return p, float(expected.min()), float((expected < 5).mean())

def categories(study):
    OUTPUT.mkdir(exist_ok=True)
    models, manifests, matrices, lineage = study.models, study.manifests, study.matrices, study.lineage
    units=study.units
    codes, names = pd.factorize(lineage)
    multipliers = dif.bootstrap_multipliers(len(names), dif.SEED)
    summary_rows, category_rows = [], []
    for benchmark in data.BENCHMARKS:
        _, questions, _, _ = study.primary(benchmark)
        questions=questions.copy()
        questions['content_category'] = category_labels(benchmark, questions, units)
        assert questions.content_category.notna().all()
        assert questions.unit_id.is_unique
        sizes = questions.content_category.value_counts()
        retained_categories = sizes[sizes >= 30].index
        selected = questions[questions.content_category.isin(retained_categories)]
        class_table = pd.crosstab(selected.content_category, selected.category)
        class_p, class_expected_min, class_small = test_table(class_table)
        large = selected[selected.category.eq('C')]
        direction_table = pd.crosstab(large.content_category, large.direction)
        direction_p, direction_expected_min, direction_small = test_table(direction_table)
        prevalence, directions = [], []
        for category, group in questions.groupby('content_category', sort=True):
            c = group[group.category.eq('C')]
            c_pct = 100*group.category.eq('C').mean()
            q_pct = 100*c.direction.eq('Qwen').mean() if len(c) else np.nan
            included = len(group) >= 30
            category_rows.append({
                'benchmark':NAMES[benchmark], 'category':category, 'items':len(group),
                'included_in_tests':included, 'A_items':int(group.category.eq('A').sum()),
                'B_items':int(group.category.eq('B').sum()), 'C_items':len(c),
                'C_pct':c_pct, 'C_favor_Qwen_items':int(c.direction.eq('Qwen').sum()),
                'C_favor_Llama_items':int(c.direction.eq('Llama').sum()),
                'C_favor_Qwen_pct':q_pct, 'C_favor_Llama_pct':100-q_pct})
            if included:
                prevalence.append(c_pct)
                if np.isfinite(q_pct):
                    directions.append(q_pct)
        summary_rows.append({
            'benchmark':NAMES[benchmark], 'categories':len(retained_categories),
            'tested_items':len(selected), 'tested_C_items':len(large),
            'C_pct_min':min(prevalence), 'C_pct_max':max(prevalence),
            'class_p':class_p, 'C_favor_Qwen_min':min(directions),
            'C_favor_Qwen_max':max(directions), 'direction_p':direction_p,
            'class_expected_min':class_expected_min, 'class_expected_fraction_lt5':class_small,
            'direction_expected_min':direction_expected_min,
            'direction_expected_fraction_lt5':direction_small})
        print(f'{benchmark}: {len(retained_categories)} categories; class p={class_p:.5g}; direction p={direction_p:.5g}', flush=True)
    summary = pd.DataFrame(summary_rows)

    detail=pd.DataFrame(category_rows)
    rows=[[r.benchmark,r.categories,f'{r.C_pct_min:.1f}--{r.C_pct_max:.1f}',tables.pvalue(r.class_p),f'{r.C_favor_Qwen_min:.1f}--{r.C_favor_Qwen_max:.1f}',tables.pvalue(r.direction_p)] for r in summary.itertuples()]
    tables.write('DIF_category_concentration.tex','tab:dif-cat-concentration',
      r'Categorical concentration of DIF. Tests compare subcategory with A/B/C classification or favored family among Category-C items, using subcategories with at least 30 retained questions. Pearson $\chi^2$ $p$-values are unadjusted. IFEval uses the first listed instruction group.',
      ['Benchmark','Cats.',r'\% C range',r'$\chi^2$ class $p$',r'\% C-favor-Qwen range',r'$\chi^2$ direction $p$'],rows)
    for name,filename in [('BBH','fig_category_dif_bbh'),('MMLU-Pro','fig_category_dif_mmlupro')]:
        plots.category_figure(name,detail,summary,filename)
    # The GPQA subdomain check is printed in the paper as prose, not a table.
    questions=study.primary('gpqa')[1].copy()
    questions['subdomain']=study.units.loc[questions.unit_id,'subdomain'].to_numpy()
    sizes=questions.subdomain.value_counts();keep=sizes[sizes>=30].index
    selected=questions[questions.subdomain.isin(keep)]
    large=selected[selected.category.eq('C')]
    proportions=selected.groupby('subdomain').category.agg(lambda s:100*s.eq('C').mean())
    direction_proportions=large.groupby('subdomain').direction.agg(lambda s:100*s.eq('Qwen').mean())
    subdomain=dict(subdomains_total=len(sizes),subdomains_tested=len(keep),tested_items=len(selected),tested_C_items=len(large),
        C_pct_min=proportions.min(),C_pct_max=proportions.max(),
        C_favor_Qwen_min=direction_proportions.min(),C_favor_Qwen_max=direction_proportions.max(),
        class_p=test_table(pd.crosstab(selected.subdomain,selected.category))[0],
        direction_p=test_table(pd.crosstab(large.subdomain,large.direction))[0])
    study.results.update(categories=summary,category_details=detail,gpqa_subdomains=subdomain)
    return summary


def position_regressions(study):
    models,manifests,matrices,lineage=study.models,study.manifests,study.matrices,study.lineage
    units=study.units
    codes,names=pd.factorize(lineage)
    multipliers=dif.bootstrap_multipliers(len(names),dif.SEED)
    coefficients,tests=[],[]
    for benchmark in ['bbh','musr']:
        _,items,_,_=study.primary(benchmark)
        items=items[items.category.eq('C')].join(units[['subtask','n_options','correct_position']],on='unit_id')
        assert items[['subtask','n_options','correct_position']].notna().all().all()
        for n_options,group in items.groupby('n_options',sort=True):
            n_options=int(n_options)
            y=group.direction.eq('Qwen').to_numpy(float)
            positions=sorted(group.correct_position.astype(int).unique())
            reference=int(positions[0])
            # A separate intercept per task; no answer-position x task interaction.
            task=pd.get_dummies(group.subtask,dtype=float)
            x0=task.to_numpy()
            terms=[]
            for position in positions[1:]:
                terms.append((group.correct_position.to_numpy()==position).astype(float))
            x=np.column_stack([x0]+terms)
            beta,covariance,iterations,error=firth_logistic(x,y)
            position_index=np.arange(x0.shape[1],x.shape[1])
            block_beta=beta[position_index]
            block_covariance=covariance[np.ix_(position_index,position_index)]
            statistic=float(block_beta@np.linalg.solve(block_covariance,block_beta))
            tests.append({'benchmark':benchmark.upper() if benchmark=='bbh' else 'MuSR',
                          'n_options':n_options,'C_items':len(group),
                          'Qwen_items':int(y.sum()),'Llama_items':int(len(y)-y.sum()),
                          'subcategories':task.shape[1], 'position_df':len(position_index),
                          'position_Wald_chi2':statistic,'position_p':chi2.sf(statistic,len(position_index)),
                          'converged':True,'iterations':iterations,'maximum_adjusted_score':error})
            for position,index in zip(positions[1:],position_index):
                se=float(np.sqrt(covariance[index,index]))
                value=float(beta[index]);low=value-1.959963984540054*se;high=value+1.959963984540054*se
                p=2*norm.sf(abs(value/se))
                selected=group[group.correct_position.eq(position)]
                coefficients.append({'benchmark':tests[-1]['benchmark'],'n_options':n_options,
                                     'position':int(position),'reference_position':reference,
                                     'C_items_at_position':len(selected),'C_items_at_reference':int(group.correct_position.eq(reference).sum()),
                                     'coefficient':value,'SE':se,'coefficient_low':low,'coefficient_high':high,
                                     'odds_ratio':np.exp(value),'OR_low':np.exp(low),'OR_high':np.exp(high),
                                     'p':p,'estimator':'Firth; approximate Wald inference'})
            print(benchmark,'options',n_options,'C items',len(group),'joint position p',tests[-1]['position_p'],flush=True)
    coef=pd.DataFrame(coefficients);joint=pd.DataFrame(tests)
    # Separate correction families: all position contrasts vs all joint tests.
    coef['q']=dif.bh(coef.p.to_numpy());joint['position_q']=dif.bh(joint.position_p.to_numpy())

    display=coef.sort_values('position').groupby(['benchmark','n_options'],as_index=False).first()
    joined=joint.merge(display,on=['benchmark','n_options'],how='left')
    rows=[[r.benchmark,r.n_options,r.C_items,f'{r.position} vs. {r.reference_position}',
        f'{r.coefficient:+.2f} [{r.coefficient_low:+.2f}, {r.coefficient_high:+.2f}]',tables.pvalue(r.q),tables.pvalue(r.position_q)] for r in joined.itertuples()]
    tables.write('DIF_position_logistic.tex','tab:dif-position-logistic',
      r'Association between correct-answer position and DIF direction among Category-C questions. Regressions control for subcategory within each option count. Firth estimation and approximate Wald inference are used. The coefficient compares the lowest two observed positions; the joint test includes all positions. BH correction is separate across the 72 coefficient tests and 13 joint tests.',
      ['Benchmark','Options','C items','Contrast',r'$\beta$ [95\% interval]','Coefficient $q$','Joint position $q$'],rows,'llrlrrr')
    study.results.update(position_coefficients=coef,position_tests=joint)
    return joined[['benchmark','n_options','C_items','coefficient','coefficient_low','coefficient_high','q','position_q']]



TWO_OPTION={'navigate':'No','web_of_lies':'No','causal_judgement':'No',
            'sports_understanding':'no','snarks':'(B)','hyperbaton':'(B)',
            'formal_fallacies':'invalid','boolean_expressions':'True'}


def answer_selections(study):
    """Equal-key-weighted actual choices, with 1,000 lineage-cluster refits."""
    models,units=study.models,study.units
    items=study.primary('bbh')[1].copy()
    items['task']=units.loc[items.unit_id,'subtask'].str.removeprefix('bbh_').to_numpy()
    items['answer']=units.loc[items.unit_id,'target'].astype(str).str.strip().to_numpy()
    breakdown=[]
    for (task,answer),g in items.groupby(['task','answer']):
        c=g[g.category.eq('C')]
        breakdown.append(dict(task=task,answer=answer,items=len(g),C_items=len(c),
            Qwen_C=int(c.direction.eq('Qwen').sum()),Llama_C=int(c.direction.eq('Llama').sum()),
            Qwen_pct_among_C=100*c.direction.eq('Qwen').mean()))
    detail=pd.DataFrame(breakdown)
    binary=detail[detail.task.isin(TWO_OPTION)]
    rows=[[tables.escape(r.task),tables.escape(r.answer),r.items,r.C_items,r.Qwen_C,r.Llama_C,tables.number(r.Qwen_pct_among_C)] for r in binary.itertuples()]
    tables.write('DIF_BBH_task_answer.tex','tab:bbh-task-answer',
      r'BBH binary tasks by correct answer. Items gives all retained questions with that answer; C gives Category-C questions. Directional counts and percentages use only Category-C questions.',
      ['Task','Correct answer','Items','C','C: Qwen','C: Llama',r'Qwen among C (\%)'],rows,'llrrrrr')
    response=pd.read_parquet(data.CURATED/'responses/bbh.parquet',columns=['model','unit_id','choice','correct'])
    response=response[response.model.isin(models.fullname)&response.unit_id.isin(items.unit_id)].copy()
    assert len(response)==len(models)*len(items) and not response.duplicated(['model','unit_id']).any()
    score=study.primary('bbh')[2].set_index('model').matching_score
    assert np.allclose(response.groupby('model').correct.mean().reindex(score.index),score)
    family=models.set_index('fullname').group
    lineage=pd.Series(study.lineage,index=models.fullname)
    response['task']=response.unit_id.map(items.set_index('unit_id').task)
    rows=[]
    for task,answer in TWO_OPTION.items():
        x=response[response.task.eq(task)].copy();ids=x.unit_id.unique()
        options={u:[str(o).strip() for o in units.at[u,'options']] for u in ids}
        assert all(len(o)==2 and answer in o for o in options.values())
        key=units.target.astype(str).str.strip()
        positions={u:o.index(answer)+1 for u,o in options.items()}
        correct={u:o.index(key.at[u])+1 for u,o in options.items()}
        assert x.choice.eq(x.unit_id.map(correct)).eq(x.correct).all()
        x['picks']=x.choice.eq(x.unit_id.map(positions))
        x['key_is_answer']=x.unit_id.map(key).eq(answer)
        by_key=x.groupby(['model','key_is_answer']).picks.mean().unstack()
        assert set(by_key.columns)=={False,True} and by_key.notna().all().all()
        per=by_key.mean(axis=1).rename('rate').to_frame()
        per['qwen']=family.reindex(per.index).eq('Qwen').astype(int)
        per['score']=score.reindex(per.index)
        rows.append(dict(task=task,answer=answer,Llama_pct=100*per.loc[per.qwen.eq(0),'rate'].mean(),
            Qwen_pct=100*per.loc[per.qwen.eq(1),'rate'].mean(),**adjusted_selection(per,lineage)))
    result=pd.DataFrame(rows)
    rows=[[tables.escape(r.task),tables.escape(r.answer),tables.number(r.Llama_pct),tables.number(r.Qwen_pct),
           f'{r.adjusted_difference_pp:+.1f} [{r.low_pp:+.1f}, {r.high_pp:+.1f}]'] for r in result.itertuples()]
    tables.write('DIF_BBH_answer_selection.tex','tab:bbh-answer-selection',
      r'Actual answer selection on retained BBH questions. Model-level rates give equal weight to both correct answers. Differences adjust for overall BBH accuracy. Intervals are 95\% percentiles from 1,000 resamples of recorded model-lineage clusters.',
      ['Task','Selected answer',r'Llama (\%)',r'Qwen (\%)',r'Adjusted gap [95\% CI]'],rows,'llrrr')
    plots.plot_selection(result)
    study.results.update(answer_selection=result,task_answers=detail)
    return result
