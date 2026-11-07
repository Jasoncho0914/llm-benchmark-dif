"""Parse handoff-format evaluation logs and retain distinct question-ID/version pairs.
Configured by data_pipeline.py. No credentials or full source logs are saved.
"""
from __future__ import annotations
import ast
import gzip
import json
import random
import threading
import time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'rawdata'/'_download'
DOCS=OUT/'docs'
TASK_PREFIXES=['bbh','gpqa','ifeval','math','mmlu_pro','musr']
GENERATIVE=('math','ifeval')
RUN_INFO_KEYS=['config','git_hash','date','transformers_version','model_name','model_source',
    'system_instruction','system_instruction_sha','chat_template','chat_template_sha',
    'n-shot','versions','task_hashes','total_evaluation_time_seconds']
LOCK=threading.Lock()
SAVED_DOCS={}

def get(url: str, token: str) -> bytes:
    """GET with gzip transfer and exponential backoff on rate limits and server errors."""
    for attempt in range(8):
        try:
            r = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}", "Accept-Encoding": "gzip"},
                stream=True,
                timeout=120,
            )
            if r.status_code == 200:
                data = r.raw.read(decode_content=False)
                return gzip.decompress(data) if r.headers.get("Content-Encoding") == "gzip" else data
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"HTTP {r.status_code}, retry {attempt}: {url.split('/resolve/')[-1][:80]}", flush=True)
                time.sleep(min(30, 2 ** attempt * 5) + random.random() * 5)
                continue
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(30, 2 ** attempt * 5))
    raise RuntimeError(f"failed after retries: {url}")


def task_of(filename: str) -> tuple[str, str, str] | None:
    """samples_leaderboard_<subtask>_<timestamp>.jsonl -> (benchmark, subtask, timestamp)."""
    # Older runs end in .json, newer in .jsonl; both are one JSON record per line.
    if not filename.startswith("samples_leaderboard_"):
        return None
    if filename.endswith(".jsonl"):
        body = filename[len("samples_leaderboard_"):-len(".jsonl")]
    elif filename.endswith(".json"):
        body = filename[len("samples_leaderboard_"):-len(".json")]
    else:
        return None
    subtask, stamp = body.rsplit("_", 1)
    for prefix in TASK_PREFIXES:
        if subtask == prefix or subtask.startswith(prefix + "_"):
            return prefix, subtask, stamp
    return None


def parse_line(line: dict, bench: str) -> dict:
    metrics = {
        k: v
        for k, v in line.items()
        if k not in {"doc_id", "doc", "target", "arguments", "resps", "filtered_resps",
                     "doc_hash", "prompt_hash", "target_hash"}
    }
    row = {
        "doc_id": line.get("doc_id"),
        "doc_hash": line.get("doc_hash"),
        "target": str(line.get("target")),
        "metrics": json.dumps(metrics),
    }
    if bench in GENERATIVE:
        resps = line.get("resps") or []
        first = resps[0] if resps else ""
        row["response"] = str(first[0] if isinstance(first, list) and first else first)
        filtered = line.get("filtered_resps") or []
        row["extracted"] = str(filtered[0] if filtered else "")
    else:
        filtered = line.get("filtered_resps") or []
        logliks = []
        for pair in filtered:
            if isinstance(pair, str):
                pair = ast.literal_eval(pair)
            if isinstance(pair, (list, tuple)):
                logliks.append(float(pair[0]))
            elif isinstance(pair, (int, float)) and not isinstance(pair, bool):
                logliks.append(float(pair))
            # (loglik, is_greedy) pairs flattened by some runs leave bare booleans; skip them.
        row["option_logliks"] = json.dumps(logliks)
        args = line.get("arguments") or {}
        texts = []
        for a in (args.values() if isinstance(args, dict) else args):
            if isinstance(a, dict):
                texts.append(str(a.get("arg_1", "")))
            elif isinstance(a, (list, tuple)) and len(a) > 1:
                texts.append(str(a[1]))
        row["option_texts"] = json.dumps(texts)
    return row


def save_docs(subtask: str, lines: list[dict]) -> None:
    """Append every question record not seen before. GPQA Diamond and Extended
    reshuffle their options between runs, so one question can have several records."""
    path = DOCS / f"{subtask}.jsonl"
    with LOCK:
        if subtask not in SAVED_DOCS:
            SAVED_DOCS[subtask] = (
                {(str(json.loads(x)["doc_id"]), json.loads(x)["doc_hash"]) for x in path.read_text().split("\n") if x.strip()}
                if path.exists() else set()
            )
        seen = SAVED_DOCS[subtask]
        new = [line for line in lines if (str(line.get("doc_id")), line.get("doc_hash")) not in seen]
        if not new:
            return
        with path.open("a") as f:
            for line in new:
                f.write(json.dumps({"doc_id": line.get("doc_id"), "doc_hash": line.get("doc_hash"),
                                    "doc": line.get("doc")}) + "\n")
                seen.add((str(line.get("doc_id")), line.get("doc_hash")))
