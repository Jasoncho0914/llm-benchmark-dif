"""The four item examples printed in the paper."""
import ast,json
import numpy as np
import pandas as pd
from .common import OUTPUT,data,dif,matching
from . import reporting

def escape(text):
    return str(text).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#').replace('$',r'\$')

def item_examples(study):
    models,manifests,matrices,lineage=study.models,study.manifests,study.matrices,study.lineage
    units=study.units
    codes,names=pd.factorize(lineage)
    multipliers=dif.bootstrap_multipliers(len(names),dif.SEED)
    focal=models.group.eq('Qwen').to_numpy()
    chosen=[]
    for b in ['bbh','musr']:
        matrix=matrices[b]
        pooled=matrix.mean(axis=0)
        eligible=(pooled>=.01)&(pooled<=.99)
        y=matrix[:,eligible]
        score=y.mean(axis=1)
        labels,ranges=matching.strata(score,focal)
        supported=labels<matching.UNMATCHED
        # The bin containing the median-score model among supported models.
        ordered=np.flatnonzero(supported)[np.argsort(score[supported],kind='stable')]
        middle=int(labels[ordered[len(ordered)//2]])
        _,items,_,_=study.primary(b)
        items=items.join(units[['subtask','target','correct_position']],on='unit_id')
        records=json.loads((data.ITEMS/f'{b}.json').read_text())
        for family in ['Qwen','Llama']:
            task='bbh_navigate' if b=='bbh' else 'musr_murder_mysteries'
            candidates=items[(items.subtask==task)&(items.category=='C')&(items.direction==family)].copy()
            if b=='bbh':
                candidates=candidates[candidates.target.eq('No' if family=='Qwen' else 'Yes')]
                candidates=candidates[candidates.source_position.map(lambda i:len(records[int(i)]['input'])<=300)]
            else:
                candidates=candidates[candidates.correct_position.eq(1 if family=='Qwen' else 2)]
            candidates=candidates.assign(magnitude=candidates.delta.abs()).sort_values(['magnitude','source_position'],ascending=[False,True])
            for index,row in candidates.iterrows():
                col=items.index.get_loc(index)
                lr=100*y[(labels==middle)&~focal,col].mean()
                qr=100*y[(labels==middle)&focal,col].mean()
                # Require a clear >=20 pp gap in the displayed median stratum.
                if (qr-lr)*(1 if family=='Qwen' else -1)>=20:
                    break
            else:
                raise ValueError(f'No suitable {b} {family} example')
            record=records[int(row.source_position)]
            if b=='bbh':
                text=record['input'].split('\nOptions:')[0]
                detail='Navigation; correct answer: '+str(record['target'])
            else:
                options=ast.literal_eval(record['choices'])
                text=record['question']+' '+ ' '.join(f'({i+1}) {v}.' for i,v in enumerate(options))
                detail='Murder mysteries; correct option: '+str(int(record['answer_index'])+1)
            ex=dict(benchmark='BBH' if b=='bbh' else 'MuSR',family=family,text=text,detail=detail,
                    delta=float(row.delta),overall_llama=100*y[~focal,col].mean(),overall_qwen=100*y[focal,col].mean(),
                    median_llama=lr,median_qwen=qr,low=ranges[middle][0],high=ranges[middle][1],
                    label=f'ex:{b}-{family.lower()}',unit_id=row.unit_id)
            chosen.append(ex)
    lines=[r'\paragraph{Individual items illustrate the answer patterns.}',
           r'Examples~\ref{ex:bbh-qwen}--\ref{ex:musr-llama} show Category-C items favoring each family on BBH and MuSR. The BBH examples have different correct yes/no answers; the MuSR examples have different correct-answer positions. MuSR excerpts show the final question and options, with the preceding narratives omitted. Accuracy within the median stratum uses the matching interval containing the median-score model among matched models.']
    for i,z in enumerate(chosen):
        if i%2==0:lines.extend([r'\noindent',r'\begin{minipage}[t]{0.485\linewidth}'])
        else:lines.extend([r'\begin{minipage}[t]{0.485\linewidth}'])
        lines.extend([
            '% Source unit: '+z['unit_id']+f'; median matching interval: {z["low"]}--{z["high"]} percent',
            r'\itemexample',
            '  {'+z['benchmark']+r' $\cdot$ favors \textbf{'+z['family']+'}}',
            '  {``'+escape(z['text'])+"''}",
            '  {$'+r'\Delta_{\mathrm{MH}}='+f'{z["delta"]:+.2f}'+r'$; '+escape(z['detail'])+'}',
            f'  {{Overall: Llama {z["overall_llama"]:.0f}\\%, Qwen {z["overall_qwen"]:.0f}\\%}}',
            f'  {{Median stratum: Llama {z["median_llama"]:.0f}\\%, Qwen {z["median_qwen"]:.0f}\\% (${z["median_qwen"]-z["median_llama"]:+.0f}$~pp)}}',
            '  {'+z['label']+'}',
            r'\end{minipage}'+(r'\hfill' if i%2==0 else ''),
        ])
        if i%2==1:lines.extend(['',r'\medskip',''])
    (OUTPUT/'DIF_item_examples.tex').write_text('\n'.join(lines).rstrip()+'\n')
    reporting.show_examples(chosen)
    study.results["examples"]=pd.DataFrame(chosen)
    return study.results["examples"]
