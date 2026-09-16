# SemEval-2027 Task 1 (RETECO) — Track 1 · CUET_AL_Masaar

System for **RETECO Track 1**: temporal retrieval (sub-track 1a) and step-wise retrieval
(sub-track 1b) over the TEMPO benchmark — 13 domains, 1,654,055 documents.
Metric: macro **nDCG@10** via `pytrec_eval`.

Task: <https://datascienceuibk.github.io/RETECO/participate.html>

**Status: Phase 0 — repo bootstrap.** No retrieval code and no measured results yet.
`PROGRESS.md` is the live status; `results/ledger.csv` is the record of every run.
No number appears in this repo unless code here produced it and it is in the ledger.

---

## Read these first, in order

1. **`CLAUDE.md`** — mission, task facts, integrity rules, the phased build plan with gates.
   The session protocol in §0 applies to humans too.
2. **`PROGRESS.md`** — current phase, what is actually verified, the one next step, blockers.
3. **`results/ledger.csv`** — one row per run.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

The test suite needs no data, no network and no credentials.

---

## Layout

```
CLAUDE.md          project rules and build plan — read first
PROGRESS.md        living status
configs/           one yaml per experiment
reteco/            the package (paths.py today; bm25/dense/fusion/rerank arrive with their phase)
eval/              cv.py, score.py (pytrec_eval wrapper), bootstrap.py
kaggle/            local -> Kaggle GPU control path — see kaggle/README.md
submit/            make_runs.py, format-checker wrapper
notes/lit/         literature notes + SUMMARY.md
results/ledger.csv one row per run (gitignored: results/runs/)
cache/             embeddings, LLM generations (gitignored)
paper/             LaTeX
tests/             offline unit tests
```

Modules are added when their phase starts, not before, so the tree stays free of empty stubs.

---

## Two machines, different jobs

| | Laptop (Git Bash) | Claude Code web session | Kaggle kernel |
|---|---|---|---|
| Internet | full | **PyPI + GitHub only** | per-kernel toggle |
| GPU | none | none | T4 / P100 |
| Role | downloads, Kaggle CLI, CPU smoke tests | writes code, runs offline tests | bulk embedding / reranking |

Anything touching HuggingFace, the RETECO site, or the Kaggle API is a **local** step —
a Claude Code web container gets 403 on all three. Claude writes the code; you run it and
paste back real output.

`reteco/paths.py` is how code stays portable across all three: it resolves the data, cache
and results roots from the environment, detecting Kaggle (`/kaggle/input` read-only,
`/kaggle/working` persistent) versus local. Import from there rather than building paths by
hand — CLAUDE.md §6 requires every script to run unchanged on Windows and Kaggle Linux.

---

## Data

Not in this repo. Download from
<https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027> into `reteco_data/`
(gitignored), giving paths like `reteco_data/track1_tempo/iota/qrels_dev.txt`.
Override the location with `RETECO_DATA`.

Start with the **IOTA** domain — 10 queries, 10,372 documents. It is the CPU smoke-test
domain for Phase 1 and small enough to iterate on locally.

---

## GPU

See **`kaggle/README.md`**. Short version: credentials live in `~/.kaggle/kaggle.json` and
never in this repo; the corpus is uploaded once as a private Kaggle Dataset and attached to
each kernel; `kaggle/kernels/smoke_gpu/` proves the path end to end in under a minute.

GPU is not needed until Phase 4.

---

## Rules that are not optional

From `CLAUDE.md` §5:

- **Dev is sacred.** Tune and select on train only; every dev evaluation is logged with a reason.
- Never use `gold_ids`, `gold_answers` or guidance annotations at inference time.
- Retrieval uses **only** the organizer corpus — no web retrieval, no external collections.
- One account, one team.
- Zero fabricated numbers, anywhere.

---

## License

Code: see `LICENSE`. Organizer corpus text is CC BY-SA 4.0; qrels and splits are CC BY 4.0 —
neither is redistributed here.
