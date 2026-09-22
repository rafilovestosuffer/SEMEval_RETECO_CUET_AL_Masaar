"""Phase 7 — rerank the fused first stage with ReasonRank-7B (H4).

GPU kernel, T4 x2. Input is a private dataset (``reteco-p7-input``) holding two things:

- ``reteco/`` — this repo's package, so the prompt/parse/window logic is the exact code that
  `tests/test_rerank.py` covers rather than a copy pasted into the kernel;
- ``runs/<domain>/run_1a_<split>.trec`` — the frozen Phase 5 step-fusion run
  (`eval/write_fused_runs.py`), top-100 per query.

It reranks the top ``DEPTH`` of every query with the authors' sliding windows (window 20,
step 10 -> two windows at depth 30) and writes the result as a full top-100 run, ranks below
``DEPTH`` untouched. Train split only: dev is sacred (§5.2).

Cost control, because a 7B model generating ~700 reasoning tokens per window is the most
expensive thing this project has run:

- **Queries are shuffled with a fixed seed and processed in chunks**, each chunk's runs and
  raw outputs written before the next starts. If the time budget ends the session early,
  what was finished is a *random sample* of the train queries and can still be scored
  paired against the fused run on the same topics.
- vLLM with tensor parallelism over both T4s, fp16 (the T4 has no bf16; the checkpoint is
  bf16). If vLLM cannot start, a transformers fallback runs instead and says so loudly.
- Local CPU smoke: ``RERANK_FAKE=1 RERANK_INPUT=<dir> RERANK_DATA=<dir> python ...`` runs the
  whole pipeline with a stand-in generator and no model, which is how this file was
  checked before it cost any quota.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "p7_out"
KAGGLE_INPUT = Path(os.environ.get("RERANK_INPUT", "/kaggle/input"))
DATA_DIR = Path(os.environ.get("RERANK_DATA", str(WORKING / "reteco_data")))
FAKE = os.environ.get("RERANK_FAKE") == "1"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
TRACK = "track1_tempo"
MODEL = "liuwenhan/reasonrank-7B"
REVISION = "3444046f1481991fd9f2021df231e6c9cb7fcef1"

SPLIT = "train"
DEPTH = 30               # rerank ranks 1-30: recall@30 0.499 vs @10 0.345 on the fused run
WINDOW, STEP = 20, 10    # the authors' BRIGHT setting
DOC_MAXLEN = 512         # tokens per passage, the authors' BRIGHT setting
REASONING_MAXLEN = 3072  # the authors' setting; max_tokens = this + 100
MAX_MODEL_LEN = 16384    # 20 x 512-token passages + prompt + 3172 output fits with room
LIMIT = int(os.environ.get("RERANK_LIMIT", "0")) or None   # queries; None = all
CHUNK = 128
SEED = 20260923
TIME_BUDGET_S = 7.6 * 3600   # stop starting chunks after this; GPU sessions cap at 9 h
TAG = "cuet_reasonrank7b"

OUT_RUNS = WORKING / "runs"
RAW = WORKING / "raw_outputs.jsonl"
REPORT = WORKING / "phase7_report.json"
report: dict[str, object] = {"model": MODEL, "revision": REVISION, "split": SPLIT,
                             "depth": DEPTH, "window": WINDOW, "step": STEP, "fake": FAKE}


def save() -> None:
    WORKING.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def die(reason: str) -> None:
    report["blocked_by"] = reason
    save()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    raise SystemExit(1)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def find_input() -> tuple[Path, Path]:
    """(package root, runs root), found by marker files so the mount layout does not matter."""
    pkg = sorted(KAGGLE_INPUT.glob("**/reteco/rerank.py"))
    runs = sorted(KAGGLE_INPUT.glob(f"**/runs/*/run_1a_{SPLIT}.trec"))
    if not (pkg and runs):
        # upload_dataset.py zips subdirectories; if Kaggle kept them zipped, unpack here
        import zipfile
        unpacked = WORKING / "input_unzipped"
        for z in sorted(KAGGLE_INPUT.glob("**/*.zip")):
            print(f"  unpacking {z}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(unpacked / z.stem)
        pkg = sorted(unpacked.glob("**/reteco/rerank.py"))
        runs = sorted(unpacked.glob(f"**/runs/*/run_1a_{SPLIT}.trec")) or sorted(
            unpacked.glob(f"**/*/run_1a_{SPLIT}.trec"))
        if runs and runs[0].parent.parent.name != "runs":
            print(f"  runs unpacked without their parent folder: {runs[0]}")
    print(f"  input root {KAGGLE_INPUT}: {len(pkg)} package copies, {len(runs)} run files")
    if not pkg:
        die(f"no reteco/rerank.py under {KAGGLE_INPUT}; attach the reteco-p7-input dataset")
    if not runs:
        die(f"no runs/*/run_1a_{SPLIT}.trec under {KAGGLE_INPUT}")
    return pkg[0].parent.parent, runs[0].parent.parent


def load_candidates(runs_root: Path, read_run) -> dict[str, dict[str, list[str]]]:
    return {d.name: read_run(d / f"run_1a_{SPLIT}.trec")
            for d in sorted(runs_root.iterdir()) if (d / f"run_1a_{SPLIT}.trec").is_file()}


def fetch_data(domains: list[str]) -> None:
    if all((DATA_DIR / TRACK / d / "documents.jsonl").is_file() for d in domains):
        print(f"  data already at {DATA_DIR}")
        return
    from huggingface_hub import snapshot_download
    patterns = [f"{TRACK}/{d}/{f}" for d in domains
                for f in ("documents.jsonl", f"examples_{SPLIT}.jsonl")]
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=patterns, max_workers=8)


def make_generator(stats: dict):
    """Returns (generate(chats) -> [str], prepare(text) -> str, backend name)."""
    if FAKE:
        def fake(chats):
            outs = []
            for chat in chats:
                n = chat[1]["content"].count("\n[")
                order = " > ".join(f"[{i}]" for i in range(n, 0, -1))
                outs.append(f"<think>fake</think> <answer>{order}</answer>")
            stats["windows"] = stats.get("windows", 0) + len(chats)
            return outs
        return fake, (lambda s: " ".join(s.strip().split()[:DOC_MAXLEN])), "fake"

    import torch
    from ftfy import fix_text
    from transformers import AutoTokenizer

    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)) / 2**30
    print(f"  {n_gpu} GPU(s): {names}, {vram:.1f} GiB total")
    report["gpus"] = names
    if vram < 24:
        die(f"{vram:.1f} GiB VRAM; a 7B model in fp16 needs two 16 GB GPUs (select T4 x2)")

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

    def prepare(text: str) -> str:
        content = fix_text(text.strip())
        return tok.convert_tokens_to_string(tok.tokenize(content)[:DOC_MAXLEN])

    def to_prompt(chat) -> str:
        return fix_text(tok.apply_chat_template(chat, tokenize=False,
                                                add_generation_prompt=True))

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, revision=REVISION, dtype="float16",
                  tensor_parallel_size=n_gpu, max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=0.90, disable_custom_all_reduce=True, seed=0)
        params = SamplingParams(temperature=0, max_tokens=REASONING_MAXLEN + 100)

        def generate(chats):
            outs = llm.generate([to_prompt(c) for c in chats], params, use_tqdm=True)
            stats["windows"] = stats.get("windows", 0) + len(outs)
            stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + sum(
                len(o.prompt_token_ids) for o in outs)
            stats["output_tokens"] = stats.get("output_tokens", 0) + sum(
                len(o.outputs[0].token_ids) for o in outs)
            stats["truncated"] = stats.get("truncated", 0) + sum(
                o.outputs[0].finish_reason == "length" for o in outs)
            return [o.outputs[0].text for o in outs]
        return generate, prepare, "vllm"
    except Exception as exc:  # noqa: BLE001 — any vLLM failure falls back, loudly
        print(f"\n  !!! vLLM unavailable ({type(exc).__name__}: {str(exc)[:400]})"
              f"\n  !!! falling back to transformers generate — much slower", flush=True)
        report["vllm_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"

    from transformers import AutoModelForCausalLM
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                 torch_dtype=torch.float16, device_map="auto")
    if next(model.parameters()).dtype != torch.float16:
        die("model did not load as fp16")

    def generate_hf(chats, batch: int = 6):
        outs = []
        for i in range(0, len(chats), batch):
            enc = tok([to_prompt(c) for c in chats[i:i + batch]], return_tensors="pt",
                      padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=REASONING_MAXLEN + 100,
                                     do_sample=False)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs += tok.batch_decode(new, skip_special_tokens=True)
            stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + int(
                enc["attention_mask"].sum())
            stats["output_tokens"] = stats.get("output_tokens", 0) + int(
                (new != tok.pad_token_id).sum())
            print(f"    hf {i + len(new)}/{len(chats)}", flush=True)
        stats["windows"] = stats.get("windows", 0) + len(chats)
        return outs
    return generate_hf, prepare, "transformers"


def main() -> int:
    started = time.monotonic()
    WORKING.mkdir(parents=True, exist_ok=True)

    section("STAGE 1 — dependencies")
    if not FAKE:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "120",
                               "--retries", "5", "vllm", "ftfy", "huggingface_hub"],
                              capture_output=True, text=True)
        print((proc.stdout or "")[-1500:], (proc.stderr or "")[-1500:])
        if proc.returncode != 0:
            print("  pip install of vllm failed; the transformers fallback will be used")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy",
                            "huggingface_hub", "accelerate"], check=False)

    pkg_root, runs_root = find_input()
    sys.path.insert(0, str(pkg_root))
    from reteco.rerank import rerank_lists, reranked_scores
    from reteco.runs import read_run, write_run

    candidates = load_candidates(runs_root, read_run)
    domains = sorted(candidates)
    print(f"  {len(domains)} domains, {sum(len(v) for v in candidates.values())} queries")

    section("STAGE 2 — queries and candidate texts")
    fetch_data(domains)
    queries: dict[str, str] = {}
    topic_domain: dict[str, str] = {}
    texts: dict[str, str] = {}
    for d in domains:
        for rec in read_jsonl(DATA_DIR / TRACK / d / f"examples_{SPLIT}.jsonl"):
            if rec["id"] in candidates[d]:
                queries[rec["id"]] = rec["query"]
                topic_domain[rec["id"]] = d
        needed = {doc for docs in candidates[d].values() for doc in docs[:DEPTH]}
        for rec in read_jsonl(DATA_DIR / TRACK / d / "documents.jsonl"):
            key = rec.get("id", rec.get("doc_id"))
            if key in needed:
                texts[key] = rec["content"]
        missing = needed - texts.keys()
        if missing:
            die(f"{d}: {len(missing)} candidate docs missing from documents.jsonl")
        print(f"  {d:<12} {len(candidates[d]):>5} queries, {len(needed):>6} candidate docs",
              flush=True)

    order = sorted(queries)
    random.Random(SEED).shuffle(order)
    if LIMIT:
        order = order[:LIMIT]
    report["queries_planned"] = len(order)

    section("STAGE 3 — model")
    stats: dict[str, int] = {}
    generate, prepare, backend = make_generator(stats)
    report["backend"] = backend
    print(f"  backend: {backend}")

    section("STAGE 4 — rerank")
    done: dict[str, list[str]] = {}
    parse_fail = 0
    rerank_started = time.monotonic()
    with RAW.open("w", encoding="utf-8") as raw:
        for c0 in range(0, len(order), CHUNK):
            if time.monotonic() - started > TIME_BUDGET_S:
                print(f"  time budget reached; stopping after {len(done)} queries")
                report["stopped_early"] = True
                break
            chunk = order[c0:c0 + CHUNK]
            cands = {t: candidates[topic_domain[t]][t] for t in chunk}
            log: list = []
            t0 = time.monotonic()
            out = rerank_lists({t: queries[t] for t in chunk}, cands, texts, generate,
                               depth=DEPTH, window=WINDOW, step=STEP, prepare=prepare, log=log)
            for entry in log:
                entry["domain"] = topic_domain[entry["topic"]]
                raw.write(json.dumps(entry) + "\n")
                parse_fail += not entry["parsed"]
            raw.flush()
            done.update(out)

            # rewrite the runs after every chunk so a killed session keeps its work
            for d in domains:
                ranked = {t: reranked_scores(v) for t, v in done.items() if topic_domain[t] == d}
                if ranked:
                    write_run(OUT_RUNS / d / f"run_1a_{SPLIT}.trec", ranked, tag=TAG)
            el = time.monotonic() - rerank_started
            report.update({"queries_done": len(done), "parse_failures": parse_fail,
                           "rerank_seconds": round(el, 1), "stats": dict(stats)})
            save()
            print(f"  chunk {c0 // CHUNK + 1}: {len(done)}/{len(order)} queries, "
                  f"{time.monotonic() - t0:.0f}s this chunk, {el / len(done):.2f}s/query, "
                  f"parse failures {parse_fail}, stats {stats}", flush=True)

    report["seconds_total"] = round(time.monotonic() - started, 1)
    save()
    section("RESULT")
    print(f"  {len(done)} queries reranked with {backend}; parse failures {parse_fail} of "
          f"{stats.get('windows', 0)} windows")
    print(f"  runs -> {OUT_RUNS}\n  raw  -> {RAW}\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
