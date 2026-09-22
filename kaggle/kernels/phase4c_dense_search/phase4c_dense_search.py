"""Phase 4c — encode the train queries and search the dense index.

GPU kernel. Attaches Phase 4b's output (the corpus embeddings) as a kernel source and
produces TREC runs for sub-tracks 1a and 1b on the **train** split only — dev is sacred
(§5.2) and nothing here needs it.

Cheap by construction: 1,211 queries and 2,762 steps against 1.65M documents already
embedded. Encoding ~4k short texts is a couple of minutes; the search is one matmul per
domain. This is the step that finally answers H1 (dense vs BM25) on our own data.

Three things carried over from earlier phases, each of which would corrupt the comparison
if dropped:

- **Queries take the `Instruct:` prompt, documents take none.** The model card specifies an
  asymmetric prompt and Phase 4b embedded documents bare. Applying the prefix on only one
  side is correct; applying it on neither, or both, is not.
- **1b queries use the official template** — base query, blank line, ``Step: <instruction>``
  with `step_instruction`, not `step`. That is what produced the organizers' published 1b
  numbers (`reteco/data.py`).
- **Ties break by corpus order.** Duplicate documents share a row and therefore score
  *exactly* equal, so ties are pervasive. `np.argsort(-scores, kind="stable")` reproduces
  the organizers' stable sort; a plain `topk` would not.

Runs are written truncated to top-100 under ``runs/<domain>/run_<sub>_<split>.trec`` --
the same layout and cut as `official_baseline.py` -- so they are directly comparable with
the Phase 2 BM25 runs and are scored by the same `eval/score_runs.py` with no changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
DATA_DIR = WORKING / "reteco_data"
RUNS = WORKING / "runs"
REPORT = WORKING / "phase4c_report.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
TRACK = "track1_tempo"
MODEL = "AQ-MedAI/Diver-Retriever-0.6B"

QUERY_PROMPT = ("Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:")
STEP_TEMPLATE = "{query}\n\nStep: {step_instruction}"

MAX_LEN = 512
BATCH = 8
TOP_K = 100          # matches official_baseline.py's written runs
SPLIT = "train"
TAG = "diver06b"

report: dict[str, object] = {"model": MODEL, "split": SPLIT, "domains": {}}


def save() -> None:
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def die(reason: str) -> None:
    report["blocked_by"] = reason
    save()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    raise SystemExit(1)


def find_embeddings() -> Path:
    """Locate Phase 4b's output among the attached inputs."""
    if not KAGGLE_INPUT.is_dir():
        die("no /kaggle/input — attach the phase4b kernel output as a source")
    for source in sorted(KAGGLE_INPUT.iterdir()):
        for base in (source / "embeddings", source):
            if base.is_dir() and any(base.glob("*/doc_index.json")):
                print(f"  embeddings at {base}")
                return base
    die(f"no embeddings under {KAGGLE_INPUT}; attach reteco-phase4b-embed-corpus")
    raise AssertionError("unreachable")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    section("STAGE 1 — dependencies and GPU")
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "120",
                           "--retries", "5", "sentence-transformers", "huggingface_hub"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"pip install failed:\n{(proc.stderr or '')[-1200:]}")

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        die("no CUDA device; enable the GPU accelerator")
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")

    emb_root = find_embeddings()

    section("STAGE 2 — query files")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=[f"{TRACK}/*/examples_{SPLIT}.jsonl",
                                      f"{TRACK}/*/steps_{SPLIT}.jsonl"], max_workers=8)
    root = DATA_DIR / TRACK
    domains = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"  {len(domains)} domains")

    section("STAGE 3 — model")
    model = SentenceTransformer(MODEL, device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAX_LEN
    if next(model.parameters()).dtype != torch.float16:
        die("model did not load as fp16")
    print("  fp16 OK")

    # The model ships its prompts in config_sentence_transformers.json, so let
    # sentence-transformers apply the "query" one rather than hand-concatenating: ST joins
    # prompt and text with no separator, and guessing at a space would change the tokens.
    use_prompt_name = "query" in (getattr(model, "prompts", None) or {})
    print(f"  query prompt via {'prompt_name=query' if use_prompt_name else 'manual prefix'}")

    def encode_queries(texts: list[str]):
        """Queries only — hence the prompt. Phase 4b embedded documents without it."""
        if use_prompt_name:
            vecs = model.encode(texts, prompt_name="query", batch_size=BATCH,
                                show_progress_bar=False, normalize_embeddings=True)
        else:
            vecs = model.encode([QUERY_PROMPT + t for t in texts], batch_size=BATCH,
                                show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    section("STAGE 4 — search")
    RUNS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    totals = {"1a": 0, "1b": 0}

    for domain in domains:
        d_emb = emb_root / domain
        if not (d_emb / "doc_index.json").is_file():
            print(f"  {domain}: no embeddings, skipping")
            continue

        meta = json.loads((d_emb / "doc_index.json").read_text(encoding="utf-8"))
        doc_ids = meta["doc_ids"]
        rows = np.asarray(meta["rows"], dtype=np.int64)
        shards = sorted(d_emb.glob("emb_*.npy"))
        vectors = np.concatenate([np.load(s) for s in shards], axis=0).astype(np.float32)
        if vectors.shape[0] != int(meta["n_unique"]):
            die(f"{domain}: {vectors.shape[0]} vectors but doc_index says {meta['n_unique']}")

        mat = torch.from_numpy(vectors).cuda().half()          # [n_unique, dim]
        rows_t = torch.from_numpy(rows).cuda()

        # ---- 1a: whole query -------------------------------------------------
        ex = list(read_jsonl(root / domain / f"examples_{SPLIT}.jsonl"))
        topics = [(r["id"], r["query"]) for r in ex]

        # ---- 1b: base query + Step: <instruction> ----------------------------
        steps_path = root / domain / f"steps_{SPLIT}.jsonl"
        steps: list[tuple[str, str]] = []
        if steps_path.is_file():
            for rec in read_jsonl(steps_path):
                for st in rec.get("steps", []):
                    steps.append((st["step_id"],
                                  STEP_TEMPLATE.format(query=rec["query"],
                                                       step_instruction=st["step_instruction"])))

        for sub, items in (("1a", topics), ("1b", steps)):
            if not items:
                continue
            ids = [i for i, _ in items]
            qv = torch.from_numpy(encode_queries([t for _, t in items])).cuda().half()
            d_runs = RUNS / domain
            d_runs.mkdir(parents=True, exist_ok=True)
            path = d_runs / f"run_{sub}_{SPLIT}.trec"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                for start in range(0, len(ids), 64):
                    sims = (qv[start:start + 64] @ mat.T).float()      # [b, n_unique]
                    per_doc = sims[:, rows_t].cpu().numpy()            # [b, n_docs]
                    for i, qid in enumerate(ids[start:start + 64]):
                        scores = per_doc[i]
                        order = np.argsort(-scores, kind="stable")[:TOP_K]
                        for rank, j in enumerate(order, start=1):
                            fh.write(f"{qid} Q0 {doc_ids[j]} {rank} "
                                     f"{float(scores[j]):.6f} {TAG}\n")
            totals[sub] += len(ids)
            print(f"  {domain:<12} {sub}  {len(ids):>5} topics -> {domain}/{path.name}", flush=True)

        del mat, rows_t, vectors
        torch.cuda.empty_cache()
        report["domains"][domain] = {"docs": len(doc_ids),  # type: ignore[index]
                                     "unique": int(meta["n_unique"])}
        save()

    report["topics"] = totals
    report["seconds"] = round(time.monotonic() - started, 1)
    save()

    section("RESULT")
    print(f"  1a {totals['1a']:,} queries, 1b {totals['1b']:,} steps "
          f"in {report['seconds']}s")
    print(f"  runs: {RUNS}")
    print("\n  Pull the runs, then score locally:")
    print("    python eval/score_runs.py --subtrack 1a --split train "
          "--runs-dir <pulled> --out cache/scores_1a_train_dense.json")
    print("    python eval/cv.py --scores cache/scores_1a_train_dense.json "
          "--compare cache/scores_1a_train.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
