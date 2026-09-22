"""Frozen pipeline, stage 1 of 3 — encode a split's queries and steps, search the dense index.

This is `phase4c_dense_search.py` with the split made a constant instead of hard-wired to
train, so the system that was selected on train runs unchanged on dev (the Phase 8 dry
run) and on test (January). Nothing here reads qrels, so it behaves the same on a split
that has none.

The pipeline, each stage checked on train first:

1. this kernel (T4, a few minutes): runs/<domain>/run_{1a,1b}_<SPLIT>.trec, top-100;
2. local CPU: ``python eval/write_fused_runs.py --split <SPLIT> ...`` — Phase 5 step fusion;
3. ``kaggle/kernels/pipeline_rerank`` (T4 x2): ReasonRank-7B over the fused top-30 (Phase 7);

then ``submit/check_format.py`` on every run before anything is submitted.

Kept identical to Phase 4c on purpose: the ``query`` prompt on queries and none on
documents, the official 1b template, fp32 scoring over fp16 embeddings, and the stable
corpus-order tie-break.

**[VERIFY] in January:** that the test split ships as ``examples_test.jsonl`` /
``steps_test.jsonl`` in the same per-domain layout. A missing file is reported, not fatal.
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
REPORT = WORKING / "pipeline_dense_report.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
TRACK = "track1_tempo"
MODEL = "AQ-MedAI/Diver-Retriever-0.6B"

QUERY_PROMPT = ("Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:")
STEP_TEMPLATE = "{query}\n\nStep: {step_instruction}"

MAX_LEN = 512
BATCH = 8
TOP_K = 100          # matches official_baseline.py's written runs
SPLIT = "dev"          # train | dev | test: the only line that changes between runs
TAG = "cuet_diver06b"

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
    """Locate Phase 4b's output among the attached inputs.

    Run 1 failed here with `kernel_sources` correctly registered on the kernel but
    /kaggle/input empty — most likely because 4b had finished only minutes earlier and its
    output was not published yet. So this prints what it actually sees before giving up:
    a wrong assumption about the mount layout and an unpublished source look identical
    from the error message alone, and they need opposite fixes.
    """
    if not KAGGLE_INPUT.is_dir():
        die("no /kaggle/input at all — attach the phase4b kernel output as a source")

    sources = sorted(KAGGLE_INPUT.iterdir())
    print(f"  /kaggle/input has {len(sources)} source(s): "
          f"{[s.name for s in sources] or 'EMPTY'}")

    # Search recursively rather than assuming a mount layout. Run 2 showed kernel output
    # arriving at /kaggle/input/notebooks/<username>/<slug>/embeddings/<domain>/ — two
    # levels deeper than a dataset mount, and not documented anywhere I could find.
    # Globbing for the marker file makes this independent of how Kaggle nests it.
    found = sorted(KAGGLE_INPUT.glob("**/doc_index.json"))
    if not found:
        die(f"no doc_index.json anywhere under {KAGGLE_INPUT} (listing above). "
            f"If the listing is EMPTY the phase4b output was not published yet — wait for "
            f"it to appear on the kernel's Output tab and re-push.")

    # Each match is <base>/<domain>/doc_index.json, so the base is two levels up.
    bases = {p.parent.parent for p in found}
    if len(bases) > 1:
        print(f"  WARNING: index files under {len(bases)} different roots: "
              f"{sorted(str(b) for b in bases)}; using the one with the most domains")
    base = max(bases, key=lambda b: len(list(b.glob("*/doc_index.json"))))
    print(f"  found {len(list(base.glob('*/doc_index.json')))} domain index files at {base}")
    return base


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

        # fp32 for the similarity matmul, not fp16. Storing the embeddings in fp16 is fine
        # -- that is a storage decision -- but producing the SCORES in fp16 quantises them
        # to a 2^-11 grid near 0.78, and run 3 measured **69.9% of retrieved documents
        # sitting in an exact-score tie group**, only ~50 distinct scores per top-100. The
        # ranking was being decided by the corpus-order tie-break rather than by the model
        # for most of the list. fp32 costs 820 MB for History's 200k unique vectors against
        # 15.6 GB available, which is nothing.
        mat = torch.from_numpy(vectors).cuda().float()         # [n_unique, dim]
        rows_t = torch.from_numpy(rows).cuda()

        # ---- 1a: whole query -------------------------------------------------
        ex_path = root / domain / f"examples_{SPLIT}.jsonl"
        if not ex_path.is_file():
            print(f"  {domain}: no examples_{SPLIT}.jsonl, no topics in this split; skipping")
            report["domains"][domain] = {"skipped": f"no examples_{SPLIT}.jsonl"}  # type: ignore[index]
            continue
        ex = list(read_jsonl(ex_path))
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
            qv = torch.from_numpy(encode_queries([t for _, t in items])).cuda().float()
            d_runs = RUNS / domain
            d_runs.mkdir(parents=True, exist_ok=True)
            path = d_runs / f"run_{sub}_{SPLIT}.trec"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                for start in range(0, len(ids), 64):
                    sims = qv[start:start + 64] @ mat.T               # [b, n_unique], fp32
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
    print("\n  Next: pull the runs, then fuse locally:")
    print(f"    python eval/write_fused_runs.py --split {SPLIT} --runs <pulled>/runs "
          f"--out cache/pipeline_{SPLIT}/fused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
