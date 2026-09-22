"""Phase 4b — embed the full Track 1 corpus with Diver-Retriever-0.6B.

GPU kernel, internet ON. Budget from the Phase 4a measurement (6,134 real tok/s on one
T4): ~11.8 GPU-hours single-GPU, ~6 h wall-clock across the two T4s Kaggle allocates.
That fits one 9-hour session, which is why this does not need a cross-session resume —
but it still writes shard-by-shard with a `.done` sentinel, because a session that dies
at hour 5 should not throw away hour 1.

Design decisions and the measurements behind them:

- **Resumable across sessions, not just within one.** Shards carry a `.done` sentinel, and
  `find_prior_shard` also looks in /kaggle/input, so attaching a previous run's output and
  re-pushing copies finished shards forward instead of recomputing them.
- **Per-domain, deduplicated within the domain.** Retrieval always runs over one domain's
  corpus, so a duplicate shared across domains is irrelevant; deduplicating within the
  domain is the saving that matters and keeps every output file self-contained. The audit
  measured 29.4% byte-identical duplication overall (History 43.7%, bitcoin 49.9%).
- **fp16, batch 8.** Both measured in Phase 4a: fp16 is 3.2x fp32, and batch 8 beat 128
  (6,073 vs 5,350 tok/s). Peak VRAM was 1.74 GB of 15.6, so the small batch costs nothing.
- **A character pre-slice before tokenising.** HuggingFace fast tokenizers encode the
  whole string and *then* truncate, so `truncation=True` does not protect against the
  1,513,968-word document in bitcoin. Slicing to PRE_SLICE characters first bounds the
  work; at ~4x what 512 tokens can consume it cannot change any embedding. It does slightly
  *raise* the dedup rate — two documents differing only past character 8,000 become
  identical, and they would have embedded identically anyway under a 512-token cap — which
  is a small bonus saving, not a loss. Measured on IOTA: 692 duplicates on full text, 720
  after the pre-slice.
- **No prompt prefix on documents.** The model card specifies an asymmetric prompt:
  queries take "Instruct: Given a web search query, retrieve relevant passages that answer
  the query\\nQuery:" and documents take an empty string. Retrieval must apply the query
  prefix on the other side; putting it here would land the two in different regions.

Outputs per domain, under /kaggle/working/embeddings/<domain>/:
    emb_<shard>.npy     float16 [n_unique_in_shard, 1024], L2-normalised
    emb_<shard>.done    sentinel, written only after the .npy is flushed
    doc_index.json      {"doc_ids": [...], "rows": [...], "n_unique": N}

``rows[i]`` is the row in the concatenated embedding matrix for ``doc_ids[i]``; every
document appears, duplicates included, so a run file can still name all 1.65M ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
DATA_DIR = WORKING / "reteco_data"
OUT = WORKING / "embeddings"
REPORT = WORKING / "phase4b_report.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
TRACK = "track1_tempo"
MODEL = "AQ-MedAI/Diver-Retriever-0.6B"

MAX_LEN = 512
BATCH = 8               # measured optimum in Phase 4a
SHARD = 50_000          # unique texts per shard -> ~100 MB fp16 at 1024 dims
PRE_SLICE = 8_000       # characters; ~4x what 512 tokens can hold, so lossless here
DIM = 1024

report: dict[str, object] = {"model": MODEL, "max_len": MAX_LEN, "batch": BATCH,
                             "domains": {}}


def save() -> None:
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def die(reason: str) -> None:
    report["blocked_by"] = reason
    save()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    raise SystemExit(1)


def install() -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "120",
                           "--retries", "5", "sentence-transformers", "huggingface_hub"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"pip install failed:\n{(proc.stderr or '')[-1200:]}")


def content_key(text: str) -> str:
    """Must match `reteco.dedup.content_key` exactly — BLAKE2b-128 over raw UTF-8."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def build_plan(doc_ids: list[str], texts: list[str]) -> tuple[list[str], list[int], int]:
    """Dedup within a domain, ordering unique texts shortest-first.

    Mirrors `reteco.dedup.build_plan` (unit-tested there, and validated against the audit's
    independent duplicate count on IOTA). Returns the unique texts, the per-document row
    index into them, and how many documents were duplicates.
    """
    first: dict[str, int] = {}
    unique: list[str] = []
    assigned: list[int] = []
    duplicates = 0
    for text in texts:
        key = content_key(text)
        idx = first.get(key)
        if idx is None:
            idx = len(unique)
            first[key] = idx
            unique.append(text)
        else:
            duplicates += 1
        assigned.append(idx)

    order = sorted(range(len(unique)), key=lambda i: len(unique[i]))
    remap = {old: new for new, old in enumerate(order)}
    return [unique[i] for i in order], [remap[i] for i in assigned], duplicates


def find_prior_shard(domain: str, shard: int) -> Path | None:
    """Look for this shard in an attached previous run's output.

    `.done` sentinels only survive *within* a session — a fresh kernel starts with an empty
    /kaggle/working — so cross-session resume needs the previous output attached as a
    dataset or kernel source. Attach it and re-push; finished shards are copied forward
    instead of recomputed. Without an attachment this returns None and everything runs,
    which is the correct behaviour for a first run.
    """
    if not KAGGLE_INPUT.is_dir():
        return None
    for source in sorted(KAGGLE_INPUT.iterdir()):
        for base in (source / "embeddings" / domain, source / domain):
            npy, done = base / f"emb_{shard:03d}.npy", base / f"emb_{shard:03d}.done"
            if npy.is_file() and done.is_file():
                return npy
    return None


def read_domain(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["id"])
            texts.append((rec.get("content") or "")[:PRE_SLICE])
    return ids, texts


def main() -> int:
    section("STAGE 1 — GPU and dependencies")
    install()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        die("no CUDA device; enable the GPU accelerator")
    n_gpu = torch.cuda.device_count()
    print(f"torch {torch.__version__}, {n_gpu} x {torch.cuda.get_device_name(0)}")
    report["gpu_count"] = n_gpu

    section("STAGE 2 — corpus")
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=[f"{TRACK}/*/documents.jsonl"], max_workers=8)
    root = DATA_DIR / TRACK
    domains = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"  {len(domains)} domains in {time.monotonic() - started:.1f}s")

    section("STAGE 3 — model")
    started = time.monotonic()
    model = SentenceTransformer(MODEL, device="cuda",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAX_LEN
    dtype = next(model.parameters()).dtype
    print(f"  loaded in {time.monotonic() - started:.1f}s, dtype {dtype}")
    if dtype != torch.float16:
        die(f"expected fp16, got {dtype} — fp32 would cost 3.2x (Phase 4a)")

    pool = None
    if n_gpu > 1:
        print(f"  starting a multi-process pool over {n_gpu} GPUs")
        try:
            pool = model.start_multi_process_pool()
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the session
            print(f"  multi-GPU pool failed ({type(exc).__name__}: {exc}); using one GPU")
            pool = None

    def encode(texts: list[str]):
        if pool is not None:
            vecs = model.encode_multi_process(texts, pool, batch_size=BATCH)
            vecs = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return (vecs / np.maximum(norms, 1e-12)).astype(np.float16)
        vecs = model.encode(texts, batch_size=BATCH, show_progress_bar=False,
                            normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float16)

    section("STAGE 4 — embed, domain by domain")
    OUT.mkdir(parents=True, exist_ok=True)
    grand_started = time.monotonic()
    total_docs = total_unique = 0

    for domain in domains:
        d_out = OUT / domain
        d_out.mkdir(parents=True, exist_ok=True)
        d_started = time.monotonic()

        ids, texts = read_domain(root / domain / "documents.jsonl")
        unique, rows, dups = build_plan(ids, texts)
        (d_out / "doc_index.json").write_text(
            json.dumps({"doc_ids": ids, "rows": rows, "n_unique": len(unique)}),
            encoding="utf-8")

        n_shards = (len(unique) + SHARD - 1) // SHARD
        print(f"\n  {domain}: {len(ids):,} docs -> {len(unique):,} unique "
              f"({dups / max(len(ids), 1):.1%} dup), {n_shards} shard(s)", flush=True)

        for s in range(n_shards):
            npy, done = d_out / f"emb_{s:03d}.npy", d_out / f"emb_{s:03d}.done"
            if done.exists() and npy.exists():
                print(f"    shard {s:03d} already done, skipping", flush=True)
                continue
            prior = find_prior_shard(domain, s)
            if prior is not None:
                shutil.copy2(prior, npy)
                done.write_text("ok", encoding="utf-8")
                print(f"    shard {s:03d} recovered from {prior.parent.parent.parent.name}",
                      flush=True)
                continue
            chunk = unique[s * SHARD:(s + 1) * SHARD]
            t0 = time.monotonic()
            vecs = encode(chunk)
            if vecs.shape != (len(chunk), DIM):
                die(f"{domain} shard {s}: got {vecs.shape}, expected {(len(chunk), DIM)}")
            if not np.isfinite(vecs.astype(np.float32)).all():
                die(f"{domain} shard {s}: non-finite embeddings under fp16")
            np.save(npy, vecs)
            os.sync() if hasattr(os, "sync") else None
            done.write_text("ok", encoding="utf-8")
            secs = time.monotonic() - t0
            print(f"    shard {s:03d}  {len(chunk):,} docs in {secs / 60:.1f} min "
                  f"({len(chunk) / max(secs, 1e-9):.1f} doc/s)", flush=True)

        elapsed = time.monotonic() - d_started
        total_docs += len(ids)
        total_unique += len(unique)
        report["domains"][domain] = {  # type: ignore[index]
            "docs": len(ids), "unique": len(unique), "duplicates": dups,
            "shards": n_shards, "seconds": round(elapsed, 1)}
        print(f"    {domain} done in {elapsed / 60:.1f} min", flush=True)
        save()

    if pool is not None:
        model.stop_multi_process_pool(pool)

    total = time.monotonic() - grand_started
    report.update({"total_docs": total_docs, "total_unique": total_unique,
                   "total_seconds": round(total, 1)})
    save()

    section("RESULT")
    print(f"  {total_docs:,} documents -> {total_unique:,} unique "
          f"({1 - total_unique / total_docs:.1%} avoided)")
    print(f"  wall clock {total / 3600:.2f} h on {n_gpu} GPU(s)")
    print(f"  embeddings: {OUT}")
    print(f"  report:     {REPORT}")
    print("\n  Next: pull the shards, then Phase 4c scores dense retrieval on train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
