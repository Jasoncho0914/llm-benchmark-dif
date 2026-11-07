"""Handoff dataset-building functions; paths and canonical order supplied by data_pipeline.py."""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

FRESH = ROOT / "rawdata" / "_download"
OUT = ROOT / "rawdata"
MC = ["bbh", "gpqa", "musr", "mmlu_pro"]
GEN = ["math", "ifeval"]
FOLDER = {benchmark: "details" for benchmark in MC + GEN}
LENGTH_NORMALIZED = {"bbh", "gpqa", "musr"}
SCORE_COLUMN = {"bbh": "BBH Raw", "gpqa": "GPQA Raw", "musr": "MUSR Raw", "mmlu_pro": "MMLU-PRO Raw",
                "math": "MATH Lvl 5 Raw", "ifeval": "IFEval Raw"}
BATCH = 2


def file_of(model: str, benchmark: str) -> Path:
    return FRESH / FOLDER[benchmark] / (model.replace("/", "__") + ".parquet")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def subtasks_of(benchmark: str) -> list[str]:
    names = [Path(p).stem for p in glob.glob(str(FRESH / "docs" / "*.jsonl"))]
    if benchmark == "bbh":
        return sorted(n for n in names if n.startswith("bbh_") and not n.startswith("bbh_fewshot"))
    if benchmark == "math":
        return sorted(n for n in names if n.startswith("math_") and n.endswith("_hard"))
    return sorted(n for n in names if n.startswith(benchmark))


# --------------------------------------------------------------------------- models

def family(arch: str, params: float) -> str:
    params = round(float(params), 2) if pd.notna(params) else np.nan
    if arch == "LlamaForCausalLM" and params == 8.03:
        return "llama3_8b"
    if arch == "Qwen2ForCausalLM" and params in (7.61, 7.62):
        return "qwen2_7b"
    if arch == "MistralForCausalLM" and params == 7.24:
        return "mistral_7b"
    return "other"


def run_dtype(info: dict) -> str | None:
    config = info.get("config")
    try:
        config = ast.literal_eval(config) if isinstance(config, str) else (config or {})
    except (ValueError, SyntaxError):
        return None
    args = config.get("model_args", "")
    match = re.search(r"dtype=([a-z0-9]+)", args)
    return match.group(1) if match else None


def build_models(targets: list[str], runs: pd.DataFrame) -> pd.DataFrame:
    contents = pd.read_parquet(FRESH / "contents.parquet")
    random30 = set(open(FRESH / "targets_random30.txt").read().split())
    rows = []
    for model in targets:
        info_path = FRESH / "run_info" / (model.replace("/", "__") + ".json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        entries = contents[contents.fullname.eq(model)]
        dtype = run_dtype(info) if info else None
        if len(entries) > 1 and dtype is not None and entries.Precision.eq(dtype).any():
            entries = entries[entries.Precision.eq(dtype)]
        e = entries.iloc[-1] if len(entries) else pd.Series(dtype=object)
        meta_path = FRESH / "model_info" / (model.replace("/", "__") + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        row = {
            "model": model,
            "family": family(e.get("Architecture"), e.get("#Params (B)")),
            "random_sample": model in random30,
            "uploader": model.split("/")[0],
            "architecture": e.get("Architecture"),
            "params_b": e.get("#Params (B)"),
            "precision": e.get("Precision"),
            "leaderboard_entries": int(contents.fullname.eq(model).sum()),
            "type": e.get("Type"),
            "weight_type": e.get("Weight type"),
            "merged": e.get("Merged"),
            "moe": e.get("MoE"),
            "official_provider": e.get("Official Providers"),
            "flagged": e.get("Flagged"),
            "chat_template": e.get("Chat Template"),
            "available_on_hub": e.get("Available on the hub"),
            "base_model": e.get("Base Model"),
            "generation": e.get("Generation"),
            "upload_date": e.get("Upload To Hub Date"),
            "submission_date": e.get("Submission Date"),
            "leaderboard_average": e.get("Average ⬆️"),
            **{f"leaderboard_{b}": e.get(col) for b, col in SCORE_COLUMN.items()},
            "evaluated_revision": meta.get("evaluated_revision"),
            "repo_status": meta.get("status", "not checked"),
            "harness_git_hash": info.get("git_hash"),
            "run_dtype": dtype,
            "chat_template_sha": info.get("chat_template_sha"),
            "system_instruction": info.get("system_instruction"),
        }
        config = meta.get("config")
        try:
            config = (ast.literal_eval(config) if isinstance(config, str) else config) or {}
        except (ValueError, SyntaxError):
            config = {}
        row.update(config_hidden=config.get("hidden_size"), config_layers=config.get("num_hidden_layers"),
                   config_vocab=config.get("vocab_size"), tokenizer_blob=meta.get("tokenizer_blob"))
        rows.append(row)
    models = pd.DataFrame(rows)
    wide = runs.pivot(index="model", columns="benchmark", values=["run", "units"])
    for b in MC + GEN:
        models[f"run_{b}"] = models.model.map(wide["run"][b]) if b in wide["run"] else None
        models[f"units_{b}"] = models.model.map(wide["units"][b]).astype("Int64") if b in wide["units"] else 0
    models["complete"] = models[[f"units_{b}" for b in MC + GEN]].fillna(0).gt(0).all(axis=1)
    return models


# --------------------------------------------------------------------------- questions

def read_docs(subtask: str) -> list[dict]:
    return [json.loads(line) for line in open(FRESH / "docs" / f"{subtask}.jsonl")]


def option_texts(benchmark: str, targets: list[str]) -> dict:
    """Option texts per version, from one log row per (subtask, doc_hash)."""
    paths = [str(file_of(m, benchmark)) for m in targets if file_of(m, benchmark).exists()]
    result = {}
    scanner = ds.dataset(paths, format="parquet").scanner(
        columns=["benchmark", "subtask", "doc_hash", "option_texts"],
        filter=ds.field("benchmark") == benchmark, batch_size=4096)
    for batch in scanner.to_batches():
        frame = batch.to_pandas().dropna(subset=["option_texts"]).drop_duplicates(["subtask", "doc_hash"])
        for task, doc_hash, texts in zip(frame.subtask, frame.doc_hash, frame.option_texts):
            key = (task, doc_hash)
            if key not in result:
                result[key] = json.loads(texts)
    return result



def build_units(benchmark: str, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    options = option_texts(benchmark, targets) if benchmark in MC else {}
    versions = []
    for subtask in subtasks_of(benchmark):
        for r in read_docs(subtask):
            doc, key = r["doc"], (subtask, r["doc_hash"])
            v = {"benchmark": benchmark, "subtask": subtask, "doc_id": str(r["doc_id"]),
                 "unit_id": f"{subtask}/{r['doc_id']}", "doc_hash": r["doc_hash"]}
            if benchmark == "bbh":
                v.update(text=doc["input"], target=str(doc["target"]), task=subtask[len("bbh_"):])
            elif benchmark == "gpqa":
                choices = [doc[f"choice{i}"] for i in range(1, 5)]
                v.update(text=doc["Question"], target=doc["answer"], question_id=f"gpqa/{doc['Record ID']}",
                         domain=doc["High-level domain"], subdomain=doc["Subdomain"], options_doc=choices)
            elif benchmark == "musr":
                v.update(text=doc["question"], context=doc["narrative"], target=str(doc["answer_choice"]),
                         template=subtask[len("musr_"):], answer_index=int(doc["answer_index"]))
            elif benchmark == "mmlu_pro":
                v.update(text=doc["question"], target=doc["answer"], subject=doc["category"], source=doc["src"],
                         question_id=f"mmlu_pro/{doc['question_id']}", options_doc=ast.literal_eval(doc["options"])
                         if isinstance(doc["options"], str) else list(doc["options"]))
            elif benchmark == "math":
                v.update(text=doc["problem"], target=str(doc["answer"]), solution=doc["solution"],
                         problem_type=doc["type"])
            elif benchmark == "ifeval":
                ids = ast.literal_eval(doc["instruction_id_list"]) if isinstance(doc["instruction_id_list"], str) \
                    else list(doc["instruction_id_list"])
                v.update(text=doc["prompt"], target="", instruction_ids=ids, n_instructions=len(ids),
                         question_id=f"ifeval/{doc['key']}")
            if benchmark in MC:
                texts = options.get(key)
                v["options"] = texts
                v["n_options"] = len(texts) if texts else None
                v["correct_position"] = correct_position(benchmark, v, texts)
            versions.append(v)
    versions = pd.DataFrame(versions)
    if "question_id" not in versions:
        versions["question_id"] = benchmark + "/" + versions.unit_id
    units = versions.drop_duplicates("unit_id", keep="first").drop(columns=["doc_hash"]).copy()
    counts = versions.groupby("unit_id").doc_hash.nunique()
    units["n_versions"] = units.unit_id.map(counts)
    units["target_changed"] = units.unit_id.map(versions.groupby("unit_id").target.nunique()) > 1
    units["text_changed"] = units.unit_id.map(versions.groupby("unit_id").text.nunique()) > 1
    # Repeated questions within the benchmark (same text, and same options where they exist).
    content = units.text.map(normalize)
    if benchmark in MC:
        content = content + " || " + units.options.map(lambda o: normalize(" | ".join(map(str, o))) if o else "")
    if benchmark == "musr":
        content = content + " || " + units.context.map(normalize)
    group = content.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:12])
    size = group.map(group.value_counts())
    units["repeat_group"] = np.where(size > 1, group, None)
    units["repeat_answers_conflict"] = units.groupby(group).target.transform("nunique").gt(1) & (size > 1)
    if benchmark == "gpqa":
        units["gpqa_main"] = units.question_id.isin(versions[versions.subtask.eq("gpqa_main")].question_id)
    return units.reset_index(drop=True), versions


def correct_position(benchmark: str, v: dict, texts: list | None) -> int | None:
    if not texts:
        return None
    if benchmark == "gpqa":
        return "ABCD".index(v["target"].strip("()")) + 1
    if benchmark == "mmlu_pro":
        return "ABCDEFGHIJ".index(v["target"]) + 1
    if benchmark == "musr":
        return v["answer_index"] + 1
    target = v["target"].strip()
    return texts.index(target) + 1 if target in texts else None


# --------------------------------------------------------------------------- responses

def mc_rows(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    ll = [np.asarray(json.loads(x), dtype=np.float32) for x in frame.option_logliks]
    texts = [json.loads(x) for x in frame.option_texts]
    if benchmark in LENGTH_NORMALIZED:
        score = [l / np.array([len(t.lstrip()) or 1 for t in tx], dtype=np.float32) for l, tx in zip(ll, texts)]
    else:
        score = ll
    metric = "acc" if benchmark == "mmlu_pro" else "acc_norm"
    return pd.DataFrame({
        "model": frame.model.to_numpy(),
        "unit_id": (frame.subtask + "/" + frame.doc_id.astype(str)).to_numpy(),
        "doc_hash": frame.doc_hash.to_numpy(),
        "correct": [bool(json.loads(m)[metric]) for m in frame.metrics],
        "choice": np.array([int(np.argmax(s)) + 1 for s in score], dtype=np.int8),
        "option_logliks": ll,
    })


def gen_rows(frame: pd.DataFrame, benchmark: str, math_verify: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = (frame.subtask + "/" + frame.doc_id.astype(str)).to_numpy()
    metrics = [json.loads(m) for m in frame.metrics]
    out = pd.DataFrame({"model": frame.model.to_numpy(), "unit_id": unit, "doc_hash": frame.doc_hash.to_numpy()})
    if benchmark == "math":
        out["old_exact_match"] = [bool(m["exact_match"]) for m in metrics]
        out["extracted"] = frame.extracted.to_numpy()
        out = out.merge(math_verify[["model", "unit_id", "math_verify"]], on=["model", "unit_id"], how="left")
        out = out.rename(columns={"math_verify": "correct"})
    else:
        out["correct"] = [bool(m["prompt_level_strict_acc"]) for m in metrics]
        out["prompt_level_loose"] = [bool(m["prompt_level_loose_acc"]) for m in metrics]
        out["inst_level_strict"] = [list(map(bool, m["inst_level_strict_acc"])) for m in metrics]
        out["inst_level_loose"] = [list(map(bool, m["inst_level_loose_acc"])) for m in metrics]
    text = pd.DataFrame({"model": out.model, "unit_id": unit, "response": frame.response.to_numpy()})
    return out, text


def build_responses(benchmark: str, targets: list[str], math_verify: pd.DataFrame | None) -> pd.DataFrame:
    (OUT / "responses").mkdir(parents=True, exist_ok=True)
    paths = [file_of(m, benchmark) for m in targets if file_of(m, benchmark).exists()]
    keep = set(subtasks_of(benchmark))
    columns = ["model", "benchmark", "subtask", "run", "doc_id", "doc_hash", "metrics"]
    columns += ["option_logliks", "option_texts"] if benchmark in MC else ["response", "extracted"]
    writer = text_writer = None
    runs = []
    for start in range(0, len(paths), BATCH):
        frame = ds.dataset([str(p) for p in paths[start:start + BATCH]], format="parquet").to_table(
            columns=columns, filter=ds.field("benchmark") == benchmark).to_pandas()
        frame = frame[frame.subtask.isin(keep)]
        if frame.empty:
            continue
        frame["doc_id"] = frame.doc_id.astype(str)
        runs.append(frame.groupby("model").agg(run=("run", "max"), runs=("run", "nunique"),
                                               units=("doc_id", "size")).reset_index())
        if benchmark in MC:
            rows = mc_rows(frame, benchmark)
            text = None
        else:
            rows, text = gen_rows(frame, benchmark, math_verify)
        table = pa.Table.from_pandas(rows, preserve_index=False)
        writer = writer or pq.ParquetWriter(OUT / "responses" / f"{benchmark}.parquet", table.schema)
        writer.write_table(table)
        if False:  # Generated text is used in memory for grading, never exported.
            (OUT / "generations").mkdir(exist_ok=True)
            t = pa.Table.from_pandas(text, preserve_index=False)
            text_writer = text_writer or pq.ParquetWriter(OUT / "generations" / f"{benchmark}.parquet", t.schema)
            text_writer.write_table(t)
        print(f"  {benchmark}: {min(start + BATCH, len(paths))}/{len(paths)} models", flush=True)
    writer.close()
    if text_writer:
        text_writer.close()
    return pd.concat(runs, ignore_index=True).assign(benchmark=benchmark)


def build_matrix(benchmark: str, units: pd.DataFrame) -> pd.DataFrame:
    (OUT / "matrices").mkdir(exist_ok=True)
    r = pd.read_parquet(OUT / "responses" / f"{benchmark}.parquet", columns=["model", "unit_id", "correct"])
    m = r.pivot(index="model", columns="unit_id", values="correct").astype("float32")
    m = m[[u for u in units.unit_id if u in m.columns]]
    m.to_parquet(OUT / "matrices" / f"{benchmark}.parquet")
    return m


def published_definition(data, benchmark, matrix):
    if benchmark in ("bbh", "musr"):
        return matrix.T.groupby(matrix.columns.str.split("/").str[0]).mean().T.mean(axis=1)
    if benchmark == "ifeval":
        r = pd.read_parquet(data / "responses" / "ifeval.parquet", columns=["model", "correct", "inst_level_strict"])
        inst = r.groupby("model").inst_level_strict.apply(lambda s: np.mean(np.concatenate(s.to_list())))
        return ((r.groupby("model").correct.mean() + inst) / 2).reindex(matrix.index)
    return matrix.mean(axis=1)
