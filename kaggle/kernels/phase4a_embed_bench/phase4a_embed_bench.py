"""Phase 4a — measure what embedding this corpus actually costs on a T4.

GPU kernel, internet ON. This is the project's FIRST GPU spend, and it exists because
every throughput figure we have below the 4B model is roofline arithmetic: no public
tokens/sec benchmark exists for modern embedding models on a T4. Budget ~1 h of the
30 GPU-h/week quota.

What it settles, in the order the answers matter:

  1. **dtype.** DIVER's cards specify bf16 and the T4 (sm75) has no bf16. HuggingFace
     defaults to fp32. A silent fp32 fallback is ~3x, and is half of why the original
     115-150 GPU-hour estimate was wrong.
  2. **Numerics.** bf16 -> fp16 narrows the exponent range, so a finite bf16 activation
     can overflow to inf. Check before a full pass, not after one.
  3. **Throughput, measured.** Three configs on the same documents, so the two library
     defaults that inflated the estimate are isolated rather than assumed:
        A  fp32 + pad-every-batch-to-512   (the naive default)
        B  fp16 + pad-to-512               (isolates dtype)
        C  fp16 + length-sorted token budget (isolates padding waste)
  4. **Batch ceiling.** Where a 16 GB T4 OOMs at 512 tokens.
  5. **T4x2 billing.** Kaggle offers two independent T4s; whether that bills quota at
     1x or 2x is undocumented and worth 2x wall-clock if it is 1x. Reported, not used.

Throughput is reported in **real tokens/sec**, never padded tokens/sec -- padding is the
thing being measured, so counting it would hide the effect.

Model facts verified from the repo on 2026-09-22 (`notes/lit/diver.md`):
last-token pooling, L2-normalised, cosine similarity, and an **asymmetric prompt** --
queries carry an "Instruct: ..." prefix, documents carry an empty one. Documents are what
we benchmark, so no prefix is applied here; the query side must add it at retrieval time.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
DATA_DIR = WORKING / "reteco_data"
REPORT = WORKING / "phase4a_report.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
TRACK = "track1_tempo"
MODEL = "AQ-MedAI/Diver-Retriever-0.6B"

SAMPLE_N = 20_000        # stratified across the 13 domains, proportional to corpus size
COMPARE_N = 2_000        # smaller sample for the slow fp32 arm, so the run fits the budget
MAX_LEN = 512
TOKEN_BUDGET = 16_384    # tokens per batch for the length-sorted arm
SEED = 20260916

report: dict[str, object] = {"model": MODEL, "max_len": MAX_LEN}


def save() -> None:
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def die(reason: str) -> None:
    report["blocked_by"] = reason
    save()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    raise SystemExit(1)


# ======================================================================== GPU ==
def probe_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        die("no CUDA device. This kernel needs the GPU accelerator enabled.")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"devices: {n}  {names}")
    print(f"capability sm{cap[0]}{cap[1]}, {total:.1f} GB, bf16_supported={bf16}")
    report.update({"torch": torch.__version__, "gpu_count": n, "gpu_names": names,
                   "capability": f"sm{cap[0]}{cap[1]}", "gpu_gb": round(total, 1),
                   "bf16_supported": bool(bf16)})
    if n > 1:
        print(f"\n  NOTE: {n} GPUs visible. This benchmark uses ONE. Whether Kaggle bills")
        print( "  the quota at 1x or 2x for a multi-GPU session is undocumented -- check the")
        print( "  quota meter before and after this run to settle it.")
    if bf16:
        print("\n  Unexpected: bf16 reported as supported. Re-check the assumption that")
        print("  sm75 lacks it before relying on fp16-only reasoning.")


def install() -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                           "--timeout", "120", "--retries", "5",
                           "sentence-transformers", "huggingface_hub"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"pip install failed:\n{(proc.stderr or '')[-1200:]}")
    import sentence_transformers, transformers  # noqa: E401
    print(f"  sentence-transformers {sentence_transformers.__version__}")
    print(f"  transformers          {transformers.__version__}")
    report["versions"] = {"sentence_transformers": sentence_transformers.__version__,
                          "transformers": transformers.__version__}


# ======================================================================= data ==
def download() -> Path:
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=[f"{TRACK}/*/documents.jsonl"], max_workers=8)
    root = DATA_DIR / TRACK
    print(f"  corpus in {time.monotonic() - started:.1f}s")
    return root


def stratified_sample(root: Path, n: int) -> list[str]:
    """Sample documents proportionally to each domain's size, preserving length mix.

    Proportional rather than equal-per-domain: the point is to estimate throughput on the
    real corpus, and the real corpus is 22% History. An equal-per-domain sample would
    over-weight the small domains and misstate the mean document length.
    """
    counts: dict[str, int] = {}
    for d in sorted(p.name for p in root.iterdir() if p.is_dir()):
        with (root / d / "documents.jsonl").open(encoding="utf-8") as fh:
            counts[d] = sum(1 for line in fh if line.strip())
    total = sum(counts.values())
    print(f"  corpus {total:,} docs across {len(counts)} domains")

    rng = random.Random(SEED)
    texts: list[str] = []
    for d, c in counts.items():
        want = max(1, round(n * c / total))
        keep = set(rng.sample(range(c), min(want, c)))
        with (root / d / "documents.jsonl").open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i in keep and line.strip():
                    texts.append(json.loads(line)["content"])
    rng.shuffle(texts)
    report["corpus_docs"] = total
    report["sample_docs"] = len(texts)
    print(f"  sampled {len(texts):,} docs")
    return texts


# ================================================================== the model ==
def load_model(dtype):
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    model = SentenceTransformer(MODEL, device="cuda",
                                model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = MAX_LEN
    actual = next(model.parameters()).dtype
    print(f"  loaded in {time.monotonic() - started:.1f}s, parameter dtype = {actual}")
    if actual != dtype:
        print(f"  WARNING: asked for {dtype}, got {actual}")
    return model, actual


def check_numerics(model, texts: list[str]) -> dict:
    """fp16 has bf16's mantissa but a far narrower exponent range -- check for overflow."""
    import torch

    emb = model.encode(texts[:256], batch_size=16, convert_to_tensor=True,
                       show_progress_bar=False, normalize_embeddings=True)
    n_inf = int(torch.isinf(emb).sum())
    n_nan = int(torch.isnan(emb).sum())
    norms = emb.float().norm(dim=1)
    out = {"dim": int(emb.shape[1]), "inf": n_inf, "nan": n_nan,
           "norm_min": round(float(norms.min()), 5),
           "norm_max": round(float(norms.max()), 5)}
    print(f"  dim {out['dim']}  inf {n_inf}  nan {n_nan}  "
          f"norms [{out['norm_min']}, {out['norm_max']}]")
    if n_inf or n_nan:
        print("  *** fp16 produced non-finite embeddings -- do NOT run the full corpus ***")
    elif abs(out["norm_min"] - 1.0) > 1e-2 or abs(out["norm_max"] - 1.0) > 1e-2:
        print("  *** normalised embeddings should have norm 1.0 -- investigate ***")
    else:
        print("  finite and unit-norm: the fp16 cast is safe on this sample")
    return out


# =================================================================== timing ==
def real_tokens(model, texts: list[str]) -> int:
    """Token count after truncation, ignoring padding -- the honest denominator."""
    tok = model.tokenizer
    total = 0
    for i in range(0, len(texts), 512):
        enc = tok(texts[i:i + 512], truncation=True, max_length=MAX_LEN)
        total += sum(len(x) for x in enc["input_ids"])
    return total


def time_encode(model, texts: list[str], batch_size: int, sort: bool, label: str) -> dict:
    """Encode and report REAL tokens/sec.

    sentence-transformers sorts by length internally by default, which is the behaviour
    we want to measure; ``sort=False`` shuffles to a fixed pathological order first so the
    padded arm actually pays the padding it is meant to demonstrate.
    """
    import torch

    batch = list(texts)
    if not sort:
        # Interleave long and short so almost every batch is padded to near max_len.
        by_len = sorted(batch, key=len)
        half = len(by_len) // 2
        batch = [x for pair in zip(by_len[:half], reversed(by_len[half:])) for x in pair]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.encode(batch, batch_size=batch_size, show_progress_bar=False,
                 normalize_embeddings=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 1e9

    toks = real_tokens(model, batch)
    tps = toks / elapsed
    dps = len(batch) / elapsed
    print(f"  {label:<34} {elapsed:7.1f}s  {dps:7.1f} doc/s  {tps:9,.0f} tok/s  "
          f"peak {peak:4.1f} GB")
    return {"label": label, "seconds": round(elapsed, 1), "docs": len(batch),
            "real_tokens": toks, "docs_per_s": round(dps, 1),
            "tokens_per_s": round(tps), "peak_gb": round(peak, 2),
            "batch_size": batch_size, "length_sorted": sort}


def project(tokens_per_s: float, label: str) -> dict:
    """Hours for the corpus at this rate, before and after deduplication."""
    full, dedup = 0.81e9, 0.26e9
    h_full, h_dedup = full / tokens_per_s / 3600, dedup / tokens_per_s / 3600
    print(f"  {label:<34} {h_full:6.1f} h untruncated   {h_dedup:6.1f} h capped+deduped")
    return {"hours_081B": round(h_full, 1), "hours_026B": round(h_dedup, 1)}


# ============================================================================ --
def main() -> int:
    section("STAGE 1 — GPU")
    probe_gpu()
    save()

    section("STAGE 2 — dependencies")
    install()
    save()

    section("STAGE 3 — corpus sample")
    texts = stratified_sample(download(), SAMPLE_N)
    lens = [len(t.split()) for t in texts]
    print(f"  sampled words/doc: median {statistics.median(lens):.0f}  "
          f"mean {statistics.fmean(lens):.0f}  max {max(lens):,}")
    report["sample_words"] = {"median": statistics.median(lens),
                              "mean": round(statistics.fmean(lens), 1), "max": max(lens)}
    save()

    section("STAGE 4 — load fp16 and check numerics")
    import torch
    model, dtype = load_model(torch.float16)
    report["loaded_dtype"] = str(dtype)
    report["numerics"] = check_numerics(model, texts)
    save()

    section("STAGE 5 — throughput: isolating the two defaults")
    print(f"  comparison sample: {COMPARE_N:,} docs, max_len {MAX_LEN}\n")
    sub = texts[:COMPARE_N]
    runs = []
    runs.append(time_encode(model, sub, 16, False, "B  fp16, padded (unsorted)"))
    runs.append(time_encode(model, sub, 16, True, "C  fp16, length-sorted"))
    del model
    torch.cuda.empty_cache()

    model32, _ = load_model(torch.float32)
    runs.append(time_encode(model32, sub, 16, False, "A  fp32, padded (the naive default)"))
    del model32
    torch.cuda.empty_cache()
    report["runs"] = runs
    save()

    section("STAGE 6 — batch size sweep (fp16, length-sorted)")
    model, _ = load_model(torch.float16)
    sweep = []
    for bs in (8, 16, 32, 64, 128):
        try:
            sweep.append(time_encode(model, sub, bs, True, f"batch {bs}"))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  batch {bs:<28} OOM")
            sweep.append({"batch_size": bs, "oom": True})
            torch.cuda.empty_cache()
            break
    report["batch_sweep"] = sweep
    save()

    best = max((r for r in sweep if "oom" not in r), key=lambda r: r["tokens_per_s"])
    section(f"STAGE 7 — scale check at batch {best['batch_size']} on {SAMPLE_N:,} docs")
    full = time_encode(model, texts, best["batch_size"], True, "full sample")
    report["scale_run"] = full
    save()

    section("STAGE 8 — projection to the real corpus")
    print("  0.81B tokens untruncated; 0.26B after a 512 cap + removing 29.4% duplicates\n")
    proj = {}
    for r in runs + [full]:
        proj[r["label"]] = project(r["tokens_per_s"], r["label"])
    report["projection"] = proj
    save()

    print(f"\n  Quota is ~30 GPU-h/week and GPU sessions cap at 9 h, so the number that")
    print( "  matters is the capped+deduped column for the fastest configuration.")
    print(f"\nreport: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
