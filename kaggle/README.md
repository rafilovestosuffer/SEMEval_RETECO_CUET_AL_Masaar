# Kaggle GPU control path

These scripts run **on Rafi's laptop**, never inside a Claude Code web session — that
container's network policy blocks `kaggle.com` (403 on CONNECT). Claude Code writes the
code; you run it and paste the output back (CLAUDE.md §3).

GPU is not needed until **Phase 4** (dense retrieval). Phases 1–3 are CPU-only. This
plumbing exists so it is already tested when Phase 4 arrives, not so Phase 4 starts early.

---

## One-time setup

```bash
# 1. Get a token: https://www.kaggle.com/settings -> Account -> API -> "Create New Token"
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Install
pip install -r requirements.txt

# 3. Check it works. Use the CLI, NOT `python -c "import kaggle"` (see the gotcha below).
kaggle --version

# 4. Confirm this repo can see the credentials (prints the username and a redacted key)
python kaggle/creds.py
```

`KAGGLE_USERNAME` + `KAGGLE_KEY` environment variables also work and take precedence over
`kaggle.json`.

### Gotcha: `kaggle/` shadows the installed `kaggle` package

This directory is named `kaggle/` because CLAUDE.md §10 says so. From the repo root, that
means `import kaggle` resolves to *this directory* (as a namespace package) instead of the
installed library, and you get a confusing `ImportError` on a perfectly good install.

Two consequences, both already handled in the code:

- There is deliberately **no `kaggle/__init__.py`**.
- `kaggle/_cli.py` shells out to the `kaggle` console script via `subprocess`, which is
  resolved through `PATH` and is immune to the shadowing.

Verify an install with `kaggle --version`, never `python -c "import kaggle"` from the repo root.

---

## Credential boundary

The API key is needed **only on your laptop**, for the local CLI. A running kernel is
already authenticated as you and reaches its data through `dataset_sources` in its
metadata. Therefore:

- No key in kernel source.
- No key in `kernel-metadata.json`.
- No key anywhere in this repository. `.gitignore` blocks `kaggle.json` and `*.key` as a
  second line of defence.
- No need for Kaggle Secrets in this workflow at all.

`creds.py` never prints, logs, or returns the key — only the username and a redacted
fingerprint (`******************ef (len=32)`), so a pasted terminal log cannot leak it.

**If a key is ever exposed** (pasted into a chat, committed, screenshotted): expire it at
<https://www.kaggle.com/settings> → Account → API → *Expire Token*, then create a new one.
Rotation is cheap; assuming it was fine is not.

---

## The loop

### 1. Smoke test — do this first

Proves the whole path works before any real job depends on it. Costs well under a minute
of the ~30 GPU-hrs/week quota.

```bash
python kaggle/push_kernel.py kaggle/kernels/smoke_gpu
python kaggle/pull_output.py --kernel reteco-smoke-gpu
```

It reports the GPU model (T4 vs P100), GPU memory, host RAM, free working disk, whether
torch sees CUDA, whether a real fp16 matmul executes, and whether bf16 is supported —
a T4 is compute capability 7.5 and a P100 is 6.0, so **neither supports bf16**; later
embedding jobs use fp16. Results land in `cache/kernel_output/reteco-smoke-gpu/` and in
`smoke_gpu_result.json`.

### 2. Upload the corpus once

```bash
python kaggle/upload_dataset.py --dir reteco_data --slug reteco-track1 \
    --title "RETECO Track 1 (TEMPO)" --dry-run     # inspect size + metadata first
python kaggle/upload_dataset.py --dir reteco_data --slug reteco-track1 \
    --title "RETECO Track 1 (TEMPO)"
```

Datasets are **private** by default (`--public` is an explicit opt-out). The corpus is
CC BY-SA 4.0, but redistributing the organizers' packaging publicly is not our call.

Why upload at all: a kernel is a fresh container every run. Re-downloading 1.65M documents
inside each one burns wall clock against the 12h session cap and makes every GPU job
depend on HuggingFace being up.

### 3. Attach it and push a real kernel

Add the dataset to `dataset_sources` in the kernel's `kernel-metadata.json`:

```json
"dataset_sources": ["rafiurrahman01/reteco-track1"]
```

Then `push_kernel.py` as above. Inside the kernel, use `reteco.paths` rather than
hard-coded paths — it resolves `/kaggle/input` and `/kaggle/working` automatically.

### 4. Version the output cache back up — this is how jobs resume

```bash
python kaggle/upload_dataset.py --dir cache/embeddings --slug reteco-cache \
    --title "RETECO embedding cache" --message "bge-m3, domains 1-6"
```

Attach `reteco-cache` as an input on the next run; the kernel skips domains/shards already
present and continues. A 12h kill then costs at most one shard, which is what CLAUDE.md §6
requires of every long job.

---

## Scripts

| Script | What it does |
|---|---|
| `creds.py` | Resolves credentials (env vars, then `~/.kaggle/kaggle.json`). Run directly to check setup. |
| `_cli.py` | Logged `subprocess` wrapper around the `kaggle` CLI. |
| `push_kernel.py` | Validates metadata, pushes, polls status to completion. `--dry-run`, `--no-wait`. |
| `pull_output.py` | Downloads kernel output into `cache/kernel_output/<slug>/` and prints the log. |
| `upload_dataset.py` | Creates or versions a private dataset. `--dry-run` reports size first. |

All take `--log-file` and `-v`. All are idempotent.

The kernel id in a committed `kernel-metadata.json` reads `USERNAME/<slug>`;
`push_kernel.py` substitutes the real username from your credentials on first push, so the
committed file carries no account name.

---

## Quota

Free Kaggle GPU is ~30 hrs/week, 12h per session, ~20 GB working disk, ~30 GB RAM.
CLAUDE.md §6 requires estimating GPU-hours *before* proposing a run. The smoke test's
reported device and memory are the inputs to that estimate — use the measured numbers,
not assumed ones.
