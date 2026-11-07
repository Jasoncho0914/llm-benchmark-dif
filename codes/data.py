"""Read rawdata and apply the current model/question selection protocol.
Functions retain the coauthor's selection logic; only paths are adapted.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / 'rawdata'
ITEMS = CURATED / 'items'
BENCHMARKS = ['bbh','gpqa','ifeval','openllm_math','mmlu_pro','musr']
CURATED_NAME = {'openllm_math':'math'}

def curated_name(benchmark: str) -> str:
    return CURATED_NAME.get(benchmark, benchmark)

def keep_units(benchmark: str, units: pd.DataFrame) -> np.ndarray:
    if benchmark == "gpqa":
        return units.subtask.eq("gpqa_main").to_numpy()
    if benchmark == "bbh":
        text = units.text.str.strip()
        return (~text.duplicated(keep=False)).to_numpy()
    if benchmark == "mmlu_pro":
        key = units.text.str.strip() + "\x1f" + units.options.map(lambda o: "\x1e".join(map(str, o)))
        return (~key.duplicated(keep="first")).to_numpy()
    return np.ones(len(units), dtype=bool)

def lineage_groups(models: pd.DataFrame):
    base = models["Base Model"].astype("string").str.strip()
    missing = base.isna() | base.isin(["", "?", "Removed", "nan"])
    base = base.astype(object)
    base.loc[missing] = "__singleton__" + models.loc[missing.to_numpy(), "fullname"].to_numpy()
    names, codes = np.unique(base.astype(str), return_inverse=True)
    return base.to_numpy(str), names, codes

def load(families=("llama3_8b", "qwen2_7b")):
    """Same return values as run_rq1.load_frozen: models, manifests, matrices,
    lineage label per model, lineage names, lineage codes."""
    if not (ITEMS / "bbh.json").exists():
        raise FileNotFoundError('Missing rawdata/items/bbh.json')
    m = pd.read_parquet(CURATED / "models.parquet")
    family_labels = {"llama3_8b":"Llama", "qwen2_7b":"Qwen", "mistral_7b":"Mistral"}
    if len(families) != 2 or not set(families).issubset(family_labels):
        raise ValueError('Select two supported model families')
    m = m[m.family.isin(families) & m.complete].sort_values("model")
    group = m.identical_responses_group
    m = m[group.isna() | ~group.duplicated(keep="first")].reset_index(drop=True)
    models = pd.DataFrame({
        "fullname": m.model, "group": m.family.map(family_labels),
        "Architecture": m.architecture, "#Params (B)": m.params_b, "Base Model": m.base_model,
        "Upload To Hub Date": m.upload_date, "Submission Date": m.submission_date,
        "Model sha": m.evaluated_revision, "Type": m.type, "Merged": m.merged,
        "Chat Template": m.chat_template, "uploader": m.uploader})
    units = pd.read_parquet(CURATED / "units.parquet")
    frames = {}
    complete = np.ones(len(models), dtype=bool)
    for benchmark in BENCHMARKS:
        name = curated_name(benchmark)
        u = units[units.benchmark.eq(name)].reset_index(drop=True)
        x = pd.read_parquet(CURATED / "matrices" / f"{name}.parquet").reindex(index=models.fullname, columns=u.unit_id)
        x = x.to_numpy(np.float32)
        frames[benchmark] = (u, x)
        complete &= ~np.isnan(x[:, keep_units(name, u)]).any(axis=1)
    models = models[complete].reset_index(drop=True)
    manifests, matrices = {}, {}
    for benchmark, (u, x) in frames.items():
        x = x[complete]
        p = x.mean(axis=0)
        keep = keep_units(curated_name(benchmark), u) & (p > 0) & (p < 1)
        positions = np.flatnonzero(keep)
        records = json.loads((ITEMS / f"{benchmark}.json").read_text())
        manifests[benchmark] = pd.DataFrame({
            "benchmark": benchmark, "source_position": positions,
            "response_column": [f"{benchmark}_{i}" for i in positions],
            "unit_id": u.unit_id.to_numpy()[positions],
            "item_sha256": [hashlib.sha256(json.dumps(records[i], sort_keys=True, ensure_ascii=False)
                                           .encode()).hexdigest() for i in positions]})
        matrices[benchmark] = x[:, positions].astype(np.uint8)
    lineage, names, codes = lineage_groups(models)
    return models, manifests, matrices, lineage, names, codes


def missing_inputs(folder=None):
    """List missing/empty input files without importing download dependencies."""
    folder=Path(folder) if folder is not None else CURATED
    required=['models.parquet','units.parquet','versions.parquet']
    for benchmark in BENCHMARKS:
        required.extend([f'matrices/{curated_name(benchmark)}.parquet',f'items/{benchmark}.json',
                         f'responses/{curated_name(benchmark)}.parquet'])
    missing=[]
    for relative in required:
        path=folder/relative
        if relative=='responses/mmlu_pro.parquet' and not path.exists():
            path=folder/'extra/mmlu_pro.parquet'  # Location in the supplied handoff.
        if not path.is_file() or path.stat().st_size==0:missing.append(relative)
    status=folder/'build_status.json'
    if status.exists() and not json.loads(status.read_text()).get('complete_cohort',False):
        missing.append('a complete model cohort (previous build was a partial test)')
    return missing


def ensure_inputs():
    """Keep complete rawdata untouched; otherwise authenticate and rebuild it."""
    missing=missing_inputs()
    if not missing:
        print('Using existing rawdata. No download is needed.')
        return CURATED
    print('rawdata is missing or incomplete; downloading and rebuilding the inputs.')
    try:
        from huggingface_hub import get_token,login
        from . import data_pipeline
    except ModuleNotFoundError as error:
        raise RuntimeError('Install codes/requirements.txt in the notebook environment before rebuilding data.') from error
    if get_token() is None:
        from getpass import getpass
        login(token=getpass('Hugging Face read token: '),add_to_git_credential=False)
    data_pipeline.run()
    if missing_inputs():raise RuntimeError('The input rebuild is incomplete; analysis was not started.')
    return CURATED
