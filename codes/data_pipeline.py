"""Download and rebuild missing rawdata from Hugging Face.

The saved local Hugging Face login is used automatically. source_manifest.json
freezes retrieval identifiers only; question text, responses and metadata are fetched from the source repositories.
Downloads are resumable. Full log JSON files are parsed in memory, never saved.
"""
from __future__ import annotations
import ast
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, get_token, hf_hub_url
from . import source_logs as logs, source_build as build

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'rawdata'
MANIFEST = Path(__file__).with_name('source_manifest.json')
BENCHMARKS = ['bbh','gpqa','musr','mmlu_pro','math','ifeval']


def _json(path, value):
    if path.name == 'rebuild_status.json':
        value=dict(value,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=1))
    temporary.replace(path)


def _paths(destination):
    destination = Path(destination).resolve()
    if destination != DESTINATION.resolve():
        raise ValueError('This rebuild writes only to reproducibility/rawdata.')
    for folder in ['_download/details','_download/docs','_download/run_info',
                   '_download/model_info','_download/repositories','_download/parts']:
        (destination/folder).mkdir(parents=True, exist_ok=True)
    fresh = destination/'_download'
    logs.OUT = fresh
    logs.DETAILS = fresh/'details'
    logs.DOCS = fresh/'docs'
    logs.RUN_INFO = fresh/'run_info'
    logs.CONTENTS = fresh/'contents.parquet'
    return destination, fresh


def check_access():
    """Test a real gated sample, without displaying or saving credentials."""
    token = get_token()
    if not token:
        raise RuntimeError('Sign in locally with huggingface_hub.login() before downloading.')
    api = HfApi(token=token)
    repo = 'open-llm-leaderboard/3rd-Degree-Burn__L-3.1-Science-Writer-8B-details'
    info = api.dataset_info(repo)
    path = next(s.rfilename for s in info.siblings if '/samples_leaderboard_' in s.rfilename)
    import requests
    response = requests.get(hf_hub_url(repo,path,repo_type='dataset',revision=info.sha),
        headers={'Authorization':f'Bearer {token}','Range':'bytes=0-1023'},timeout=60)
    response.raise_for_status()
    print('Hugging Face authentication and restricted evaluation-log access verified.', flush=True)
    return api


def _repository(api, repo, destination):
    path = destination/'_download/repositories'/f"{repo.split('/')[-1]}.json"
    if path.exists():
        return json.loads(path.read_text())
    frozen=json.loads(MANIFEST.read_text()).get('repository_revisions',{}).get(repo)
    info = api.dataset_info(repo,revision=frozen)
    record = {'repo':repo,'revision':info.sha,'files':[s.rfilename for s in info.siblings]}
    _json(path,record)
    return record


def _metadata(record, api, token, fresh):
    """Read configuration at the evaluated revision; no weights or tokenizer downloads."""
    name = record['model']; path=fresh/'model_info'/f"{name.replace('/','__')}.json"
    if path.exists():
        return
    out = {'model':name,'evaluated_revision':record['revision']}
    try:
        info=api.model_info(name,revision=record['revision'],files_metadata=True)
        out['revision_found']=True
    except Exception as error:
        code=getattr(getattr(error,'response',None),'status_code',None)
        if code not in (401,403,404):
            raise
        try:
            info=api.model_info(name,files_metadata=True)
            out['revision_found']=False
        except Exception as fallback_error:
            fallback_code=getattr(getattr(fallback_error,'response',None),'status_code',None)
            if fallback_code not in (401,403,404):raise
            out.update(status=f'unavailable: HTTP {fallback_code}',revision_found=False)
            _json(path,out)
            return
    out.update(sha=info.sha,status='ok',tokenizer_blob=None)
    for sibling in info.siblings or []:
        if sibling.rfilename=='tokenizer.json':
            out['tokenizer_blob']=sibling.lfs.sha256 if sibling.lfs else sibling.blob_id
    import requests
    r=requests.get(f'https://huggingface.co/{name}/resolve/{info.sha}/config.json',
                   headers={'Authorization':f'Bearer {token}'},timeout=60)
    if r.status_code==200:
        out['config']=r.json()
    elif r.status_code in (401,403,404):
        out['config']={'status':r.status_code}
    else:
        r.raise_for_status()
    _json(path,out)


_HTTP_LOCAL = threading.local()


def _stream_task(url,token,model,benchmark,subtask,stamp):
    """Stream JSON lines so large prompts do not accumulate in memory."""
    import requests
    if not hasattr(_HTTP_LOCAL,'session'):
        _HTTP_LOCAL.session=requests.Session()
    for attempt in range(5):
        rows=[];documents=[]
        try:
            with _HTTP_LOCAL.session.get(url,headers={'Authorization':f'Bearer {token}'},
                    timeout=(30,120),stream=True) as response:
                if response.status_code in (429,500,502,503,504):
                    time.sleep(min(30,2**attempt*2));continue
                response.raise_for_status()
                # Delimiter is a literal newline, not Unicode line separators inside documents.
                for line in response.iter_lines(chunk_size=65536,delimiter=b'\n'):
                    if not line.strip():continue
                    record=json.loads(line)
                    documents.append({'doc_id':record.get('doc_id'),'doc_hash':record.get('doc_hash'),'doc':record.get('doc')})
                    rows.append(dict(model=model,benchmark=benchmark,subtask=subtask,run=stamp,
                                     **logs.parse_line(record,benchmark)))
                    if len(documents)>=128:
                        logs.save_docs(subtask,documents);documents=[]
                if documents:logs.save_docs(subtask,documents)
            return rows
        except (requests.ConnectionError,requests.Timeout):
            if attempt==4:raise
            time.sleep(min(30,2**attempt*2))
    raise RuntimeError(f'Failed after retries: {model}/{subtask}')


def _download_model(record, api, token, destination, fresh):
    name=record['model']; stem=name.replace('/','__')
    if shutil.disk_usage(destination).free < 2*1024**3:
        raise RuntimeError('Less than 2 GiB of disk space remains; download stopped to preserve free space.')
    target=fresh/'details'/f'{stem}.parquet'
    if not target.exists():
        repo=f'open-llm-leaderboard/{stem}-details'
        try:
            repository=_repository(api,repo,destination)
        except Exception as error:
            # Missing repositories stay explicit; they are never silently omitted.
            raise RuntimeError(f'Cannot retrieve evaluation repository for {name}: {type(error).__name__}') from error
        newest={}
        for path in repository['files']:
            parsed=logs.task_of(path.split('/')[-1])
            if parsed is None:
                continue
            benchmark,subtask,stamp=parsed
            cutoff=record['runs'].get(benchmark)
            if not cutoff or stamp>cutoff:
                continue
            if subtask not in newest or stamp>newest[subtask][2]:
                newest[subtask]=(path,benchmark,stamp)
        required={b for b,stamp in record['runs'].items() if stamp}
        found={entry[1] for entry in newest.values()}
        if required-found:
            raise RuntimeError(f'{name}: frozen evaluation files missing for {sorted(required-found)}')
        cutoff=max((stamp for stamp in record['runs'].values() if stamp),default='')
        results=sorted(p for p in repository['files'] if p.split('/')[-1].startswith('results_')
                       and p.split('/')[-1][len('results_'):].removesuffix('.json')<=cutoff)
        # Use the most recent results available by the frozen evaluation date.

        if results and not (fresh/'run_info'/f'{stem}.json').exists():
            path=results[-1]
            info=json.loads(logs.get(hf_hub_url(repo,path,repo_type='dataset',revision=repository['revision']),token))
            slim={k:info.get(k) for k in logs.RUN_INFO_KEYS}
            slim.update(results_file=path,dataset_revision=repository['revision'])
            _json(fresh/'run_info'/f'{stem}.json',slim)
        parts=fresh/'parts'/stem;parts.mkdir(exist_ok=True)
        for subtask,(path,benchmark,stamp) in sorted(newest.items()):
            part=parts/f'{subtask}.parquet'
            if part.exists():
                continue
            rows=_stream_task(hf_hub_url(repo,path,repo_type='dataset',revision=repository['revision']),
                              token,name,benchmark,subtask,stamp)
            if not rows:
                raise RuntimeError(f'Empty evaluation file: {name}/{subtask}')
            frame=pd.DataFrame(rows)
            # Give every per-model parquet the same columns, including missing benchmarks.
            for column in ['response','extracted','option_logliks','option_texts']:
                if column not in frame:
                    frame[column]=None
            temporary=part.with_suffix('.tmp');frame.to_parquet(temporary,index=False,compression='zstd');temporary.replace(part)
        frame=pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('*.parquet'))],ignore_index=True)
        temporary=target.with_suffix('.tmp');frame.to_parquet(temporary,index=False,compression='zstd');temporary.replace(target)
        shutil.rmtree(parts)
    _metadata(record,api,token,fresh)
    return {'model':name,'status':'ok'}


def download(destination=DESTINATION, workers=4, limit=None, grade_while_downloading=False):
    """Download the frozen cohort. limit is a smoke test, never a full rebuild."""
    destination,fresh=_paths(destination)
    api=check_access();token=get_token()
    manifest=json.loads(MANIFEST.read_text())
    records=manifest['models'][:limit]
    contents_repo=_repository(api,'open-llm-leaderboard/contents',destination)
    if not (fresh/'contents.parquet').exists():
        contents_path='data/train-00000-of-00001.parquet'
        (fresh/'contents.parquet').write_bytes(logs.get(hf_hub_url('open-llm-leaderboard/contents',
            contents_path,repo_type='dataset',revision=contents_repo['revision']),token))
    (fresh/'targets_all.txt').write_text('\n'.join(r['model'] for r in manifest['models'])+'\n')
    (fresh/'targets_random30.txt').write_text('\n'.join(r['model'] for r in manifest['models'] if r['random_sample'])+'\n')
    report_path=fresh/'download_status.json'
    status=json.loads(report_path.read_text()) if report_path.exists() else {}
    errors=[];start=time.perf_counter()
    grade_pool=None;grade_futures=[]
    if grade_while_downloading:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from importlib.metadata import version
        if version('math-verify')!='0.5.2':raise RuntimeError('math-verify==0.5.2 is required.')
        (fresh/'math_verify').mkdir(exist_ok=True)
        grade_pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
            initializer=_math_worker_init,initargs=(str(fresh),))
    print(f'Downloading/checking {len(records)} models; completed files are reused.',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_download_model,r,api,token,destination,fresh):r['model'] for r in records}
        for count,future in enumerate(as_completed(futures),1):
            model=futures[future]
            try:
                result=future.result()
                if grade_pool is not None:
                    grade_futures.append(grade_pool.submit(_math_worker,model))
            except Exception as error:
                result={'model':model,'status':'error','reason':str(error)[:400]}
                errors.append(result)
            status[model]=result
            _json(report_path,status)
            if count%10==0 or result['status']=='error' or count==len(records):
                print(f'{count}/{len(records)} checked; {len(errors)} errors; {(time.perf_counter()-start)/60:.1f} min',flush=True)
                if result['status']=='error':print(model,result['reason'],flush=True)
    # Freeze the exact repository commits for subsequent independent rebuilds.
    manifest['repository_revisions']={json.loads(p.read_text())['repo']:json.loads(p.read_text())['revision']
        for p in (fresh/'repositories').glob('*.json')}
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n')
    if grade_pool is not None:
        try:
            print('Finishing MATH grading for downloaded models.',flush=True)
            for future in as_completed(grade_futures):future.result()
        finally:
            grade_pool.shutdown(wait=True)
    if errors:
        raise RuntimeError(f'{len(errors)} models could not be downloaded; see rawdata/_download/download_status.json. Rerun to retry.')
    return status


def _ordered_docs(manifest,fresh):
    """Restore original canonical versions/order using retrieval IDs only."""
    bytask={}
    for entry in manifest['units']:
        task=entry['unit_id'].rsplit('/',1)[0]
        bytask.setdefault(task,[]).append(entry)
    for task,entries in bytask.items():
        path=fresh/'docs'/f'{task}.jsonl'
        rows=[json.loads(x) for x in path.read_text().split('\n') if x.strip()]
        bykey={(f"{task}/{r['doc_id']}",r['doc_hash']):r for r in rows}
        keys=[(e['unit_id'],e['doc_hash']) for e in entries]
        missing=[key for key in keys if key not in bykey]
        if missing:
            raise RuntimeError(f'{task}: {len(missing)} canonical question/version pairs absent from downloaded logs. No baseline data substituted.')
        chosen=[bykey[key] for key in keys]
        selected=set(keys)
        remaining=[bykey[key] for key in sorted(bykey) if key not in selected]
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen+remaining))


_MATH_SOLUTIONS = {}
_MATH_PARSED_GOLD = {}
_MATH_FRESH = None


def _math_worker_init(fresh):
    global _MATH_SOLUTIONS, _MATH_PARSED_GOLD, _MATH_FRESH
    _MATH_FRESH = Path(fresh)
    _MATH_PARSED_GOLD = {}
    _MATH_SOLUTIONS = {}
    for p in (_MATH_FRESH/'docs').glob('math_*_hard.jsonl'):
        for line in p.read_text().split('\n'):
            if line.strip():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue  # An unrelated downloader may be appending its final line.
                _MATH_SOLUTIONS[row['doc_hash']]=row['doc']['solution']


def _math_worker(model):
    from math_verify import LatexExtractionConfig,parse,verify
    fresh=_MATH_FRESH
    out=fresh/'math_verify'/f"{model.replace('/','__')}.parquet"
    if out.exists():return model
    frame=pd.read_parquet(fresh/'details'/f"{model.replace('/','__')}.parquet",
        columns=['model','benchmark','subtask','doc_id','doc_hash','response'])
    frame=frame[frame.benchmark.eq('math') & frame.subtask.str.endswith('_hard')].copy()
    verdicts=[]
    for task,h,response in zip(frame.subtask,frame.doc_hash,frame.response):
        if h not in _MATH_SOLUTIONS:
            # Later downloads can introduce new evaluated question versions.
            for line in (fresh/'docs'/f'{task}.jsonl').read_text().split('\n'):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                _MATH_SOLUTIONS[row['doc_hash']]=row['doc']['solution']
        if h not in _MATH_PARSED_GOLD:
            _MATH_PARSED_GOLD[h]=parse(_MATH_SOLUTIONS[h],[LatexExtractionConfig()])
        try:verdicts.append(bool(verify(_MATH_PARSED_GOLD[h],parse(response or ''))))
        except Exception:verdicts.append(False)
    result=frame[['model','subtask','doc_id','doc_hash']].assign(math_verify=verdicts)
    temporary=out.with_suffix('.tmp');result.to_parquet(temporary,index=False);temporary.replace(out)
    return model


def _grade_math(targets,fresh,destination,workers=4):
    from importlib.metadata import version
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if version('math-verify')!='0.5.2':
        raise RuntimeError('This rebuild requires math-verify==0.5.2.')
    folder=fresh/'math_verify';folder.mkdir(exist_ok=True)
    todo=[m for m in targets if not (folder/f"{m.replace('/','__')}.parquet").exists()]
    print(f'MATH grading: {len(todo)} models remaining.',flush=True)
    # Separate processes let Math-Verify use its normal per-call signal timeout.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=_math_worker_init,initargs=(str(fresh),)) as pool:
        for count,model in enumerate(pool.map(_math_worker,todo,chunksize=1),1):
            if count%10==0 or count==len(todo):print(f'MATH grading {count}/{len(todo)} new models',flush=True)
    frame=pd.concat([pd.read_parquet(folder/f"{m.replace('/','__')}.parquet") for m in targets],ignore_index=True)
    frame['unit_id']=frame.subtask+'/'+frame.doc_id.astype(str)
    return frame


def rebuild(destination=DESTINATION, limit=None):
    """Build data tables entirely from downloaded logs and regrade MATH responses."""
    destination,fresh=_paths(destination)
    manifest=json.loads(MANIFEST.read_text());records=manifest['models'][:limit]
    targets=[r['model'] for r in records]
    missing=[m for m in targets if not (fresh/'details'/f"{m.replace('/','__')}.parquet").exists()]
    if missing:raise RuntimeError(f'{len(missing)} model downloads missing; run download() first.')
    if limit is None:_ordered_docs(manifest,fresh)
    build.FRESH=fresh;build.OUT=destination;build.FOLDER={b:'details' for b in BENCHMARKS}
    math=_grade_math(targets,fresh,destination)
    units_all=[];versions_all=[];runs=[];matrices={}
    order={e['unit_id']:i for i,e in enumerate(manifest['units'])}
    for benchmark in BENCHMARKS:
        print('Building',benchmark,flush=True)
        units,versions=build.build_units(benchmark,targets)
        units=units.sort_values('unit_id',key=lambda s:s.map(order)).reset_index(drop=True)
        units_all.append(units)
        versions_all.append(versions[['benchmark','unit_id','doc_hash','target']+[c for c in ['options','correct_position'] if c in versions]])
        runs.append(build.build_responses(benchmark,targets,math if benchmark=='math' else None))
        matrices[benchmark]=build.build_matrix(benchmark,units)
        records_by_unit={}
        for task in units.subtask.unique():
            for row in build.read_docs(task):records_by_unit.setdefault(f"{task}/{row['doc_id']}",row['doc'])
        documents=[]
        for unit in units.unit_id:
            doc=dict(records_by_unit[unit])
            for field in {'options','instruction_id_list','kwargs'} & doc.keys():
                if isinstance(doc[field],str):doc[field]=ast.literal_eval(doc[field])
            documents.append(doc)
        (destination/'items').mkdir(exist_ok=True)
        item_name='openllm_math' if benchmark=='math' else benchmark
        (destination/'items'/f'{item_name}.json').write_text(json.dumps(documents,ensure_ascii=False))
    runs=pd.concat(runs,ignore_index=True);models=build.build_models(targets,runs)
    signature=pd.concat([matrices[b].reindex(models.model) for b in BENCHMARKS],axis=1).fillna(-1)
    digest=signature.apply(lambda row:hashlib.sha1(row.to_numpy().tobytes()).hexdigest()[:12],axis=1)
    size=digest.map(digest.value_counts())
    models['identical_responses_group']=models.model.map(digest.where(size>1))
    models['runs_per_benchmark_max']=models.model.map(runs.groupby('model').runs.max())
    for b in BENCHMARKS:
        estimate=build.published_definition(destination,b,matrices[b])
        published=models.set_index('model')[f'leaderboard_{b}']
        models[f'matches_published_{b}']=models.model.map((estimate-published.reindex(estimate.index)).abs()<1e-6).fillna(False).astype(bool)
    models.to_parquet(destination/'models.parquet',index=False)
    pd.concat(units_all,ignore_index=True).to_parquet(destination/'units.parquet',index=False)
    pd.concat(versions_all,ignore_index=True).to_parquet(destination/'versions.parquet',index=False)
    _json(destination/'build_status.json',{'complete_cohort':limit is None,'models':len(models),'math_verify':'0.5.2'})
    print('Downloaded inputs rebuilt in rawdata.',flush=True)



def validate_inputs():
    """Check cohort, question IDs and alignment; this is not a completed paper replication."""
    from . import data
    missing=data.missing_inputs()
    if missing:raise RuntimeError('Missing rebuilt inputs: '+', '.join(missing))
    manifest=json.loads(MANIFEST.read_text())
    models=pd.read_parquet(DESTINATION/'models.parquet')
    units=pd.read_parquet(DESTINATION/'units.parquet')
    if set(models.model)!={r['model'] for r in manifest['models']}:
        raise RuntimeError('Rebuilt model list differs from the frozen source list.')
    if not units.unit_id.is_unique or set(units.unit_id)!={r['unit_id'] for r in manifest['units']}:
        raise RuntimeError('Rebuilt question IDs differ from the frozen source list.')
    for benchmark in BENCHMARKS:
        selected=units[units.benchmark.eq(benchmark)]
        matrix=pd.read_parquet(DESTINATION/'matrices'/f'{benchmark}.parquet')
        if list(matrix.columns)!=selected.unit_id.tolist():
            raise RuntimeError(f'{benchmark}: matrix columns do not match question-record order.')
        item_name='openllm_math' if benchmark=='math' else benchmark
        items=json.loads((DESTINATION/'items'/f'{item_name}.json').read_text())
        if len(items)!=len(selected):raise RuntimeError(f'{benchmark}: question records are not aligned.')
    return {'models':len(models),'questions':len(units),'input_structure_checked':True}


def cleanup_downloads():
    """Remove temporary source records after a successful full build."""
    fresh=DESTINATION/'_download'
    records={json.loads(p.read_text())['repo']:json.loads(p.read_text())['revision']
             for p in (fresh/'repositories').glob('*.json')}
    _json(DESTINATION/'download_provenance.json',{'repositories':records,
        'source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest()})
    if fresh.exists():shutil.rmtree(fresh)


def run(workers=4,keep_downloads=False):
    """Use complete inputs as-is; otherwise resume downloading and rebuild rawdata."""
    from . import data
    if not data.missing_inputs():
        print('Using existing rawdata; no download or changes required.',flush=True)
        return {'source':'existing rawdata'}
    stage='download and MATH grading'
    try:
        _json(DESTINATION/'rebuild_status.json',{'stage':stage,'status':'running'})
        download(workers=workers,grade_while_downloading=True)
        stage='build'
        _json(DESTINATION/'rebuild_status.json',{'stage':stage,'status':'running'})
        rebuild()
        stage='check input structure'
        result=validate_inputs()
        if not keep_downloads:cleanup_downloads()
        _json(DESTINATION/'rebuild_status.json',{'stage':'complete','status':'built',**result})
        print('rawdata is ready. Continue the notebook to run and validate the paper analyses.',flush=True)
        return result
    except Exception as error:
        _json(DESTINATION/'rebuild_status.json',{'stage':stage,'status':'error','reason':str(error)[:1000]})
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--keep-downloads',action='store_true')
    args=parser.parse_args()
    run(workers=args.workers,keep_downloads=args.keep_downloads)
