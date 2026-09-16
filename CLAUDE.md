# CLAUDE.md — SemEval-2027 Task 1 (RETECO) · Rafi / CUET_AL_Masaar

> Put this file at the repo root. Claude Code reads it at the start of every session.
> Written 16 Sept 2026. Facts about the task were taken from the official RETECO site on that date;
> anything marked **[VERIFY]** must be re-checked against the official source before relying on it.

---

## 0. Session protocol (do this every session, in order)

1. Read `CLAUDE.md` (this file), then `PROGRESS.md`, then the last 20 lines of `results/ledger.csv`.
2. State in 3–5 lines: current phase, last verified result, the ONE next step you propose.
3. Wait for Rafi's go-ahead before any long run, any download > 1 GB, or any dev-split evaluation.
4. Do the step. Smoke-test locally on CPU with a tiny domain before anything goes to Kaggle GPU.
5. End of session: update `PROGRESS.md` (what changed, what's verified, what's next) and append to the ledger.

Never report a number that was not produced by code in this repo and logged in the ledger.

---

## 1. Mission

Build a strong, reproducible system for RETECO **Track 1 (Sub-tracks 1a + 1b)**, place as high as honestly possible,
and write a SemEval system-description paper whose value is a real finding about temporal retrieval — not rank.

Honest framing: no prompt guarantees 1st place. What wins is (a) correct wiring, (b) disciplined validation,
(c) many cheap validated experiments, (d) avoiding wasted GPU hours. Optimize for those.

Secondary goal: every component and ablation should be reusable for Rafi's **Bangla Legal RAG** project
(multi-hop retrieval across amended statutes = temporal chaining). Design modules to be language-agnostic.

---

## 2. Who you're working with

- **Rafi (Rafiur Rahman)** — 2nd-year Mechanical Engineering undergrad, CUET, Bangladesh. Solo self-directed AI/ML researcher.
  Kaggle: `rafiurrahman01`. GitHub: `rafilovestosuffer`. Team name: `CUET_AL_Masaar`.
- Track record: 1st place (public + private LB) IEEE CS CUET ML Contest 2.0; top-15 finalist IUT ICT Fest Datathon 2026
  (Bengali hallucination detection, 0.937 F1).
- Goal: MS/PhD abroad; ACL Anthology papers via shared tasks are a core strategy.
- Parallel commitments (they limit time): semester finals (to 20 Sept 2026), NuNO paper (deadline 30 Sept 2026),
  tutoring, SemEval-2027 **Task 6 MMCultureQA** (primary SemEval task), BD-BrowseComp (ACL 2027, January ARR cycle),
  several journal papers. RETECO is **secondary** — respect that when proposing scope.

---

## 3. How to work with Rafi (mandatory)

- English by default. Bangla/easy Bangla with real-world analogies ONLY when he asks.
- He writes tersely, with typos or Banglish. Infer intent; don't make him re-explain.
- Give ONE definitive recommendation, with reasoning. No menus of hedged options. Don't reverse later without new evidence.
- Push back directly when he's wrong or a plan is weak.
- Iterative: propose one step → he approves/runs → he pastes real results → you analyze. Don't get ahead of verified results.
- Code must be complete and runnable. No partial snippets, no `...` placeholders.
- When editing existing files: change ONLY what was asked. No unrequested refactors, renames, reformatting, or "improvements".
  If you notice a problem elsewhere, mention it in one line; don't fix it unasked.
- Explain concepts beyond CNN/RNN/LSTM/ViT/attention basics from the ground up (e.g. dense retrieval, nDCG, RRF, cross-encoders).
- Deep-research mode by default for design questions: read the actual papers/repos, cite links, compare approaches, flag limitations.
- Paper/deliverable writing: plain, factual language. No rhetorical contrast with other teams, no hype, no cleverness for its own sake.
- Never fabricate citations, metrics, model names, or paper claims. If unsure a paper/model exists or says X, check it
  (alphaXiv / Hugging Face / arXiv MCP tools) or mark it **[UNVERIFIED]**.

---

## 4. Task facts (from official site, 16 Sept 2026)

Official links:
- Guide: https://datascienceuibk.github.io/RETECO/participate.html
- Data page: https://datascienceuibk.github.io/RETECO/data.html
- Evaluation: https://datascienceuibk.github.io/RETECO/evaluation.html
- Dataset: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Starter kit: https://github.com/DataScienceUIBK/RETECO/tree/main/starter_kit
- Source benchmarks: TEMPO https://github.com/tempo-bench/Tempo · RECOR https://github.com/RECOR-Benchmark/RECOR
- Lead organizer contact: abdelrahman.abdallah@uibk.ac.at (mailing list: semeval-2027-reteco)

Structure:
- Track 1 (TEMPO, 13 domains, 1,654,055 docs): **1a** temporal retrieval (rank docs per query), **1b** step-wise retrieval (rank docs per decomposed step).
- Track 2 (RECOR, 11 domains, 507,141 docs): 2a conversational retrieval, 2b gold-passage generation, 2c full conversational RAG.
- Any subset of sub-tracks allowed; un-entered sub-tracks are unranked (not zero). Missing topics within an entered sub-track score 0.
- All data is English. Corpus text CC BY-SA 4.0; qrels/splits CC BY 4.0.

Splits (gold labels public for both):
| Track | train | dev |
|---|---|---|
| 1 queries / steps | 1,211 / 2,762 | 519 / 1,214 |
| 2 conversations / turns | 496 / 2,113 | 211 / 858 |

- Corpus is never split: retrieval always runs over the full domain corpus.
- train+dev = the entire public TEMPO/RECOR benchmarks → the SemEval test set is new and unseen.
  **[VERIFY]** whether test queries come from the same 13 domains/corpora. Ask on the mailing list if not stated.

Metric: **nDCG@10**, `pytrec_eval` `ndcg_cut_10`, macro-averaged over topics. `scorer.py` adds temporal precision/coverage diagnostics (not ranking).

Run format (TREC, 6 cols): `topic Q0 doc_id rank score tag`. Topic ids: 1a = `id` (e.g. `124973_5`); 1b = `step_id` (e.g. `124973_5_step1`).
Always run `format_checker.py` with `--qrels` and `--corpus` before scoring.

Files per Track 1 domain: `documents.jsonl` (id/doc_id, content), `examples_{train,dev}.jsonl` (id, query, gold_ids, gold_answers),
`steps_{train,dev}.jsonl` (id, query, steps[step_id, step, step_instruction, gold_ids]), `guidance_*.jsonl` (query_guidance,
gold_passage_annotations), `qrels_*.txt`, `qrels_steps_*.txt`, `split_manifest.json`.
Path example: `reteco_data/track1_tempo/iota/qrels_dev.txt`.

Official BM25 baseline (Lucene analyzer, gensim LuceneBM25Model k1=0.9 b=0.4), macro nDCG@10:
| Sub-track | train | dev |
|---|---|---|
| 1a whole query | 0.0879 | 0.0967 |
| 1b query + step instruction | 0.0852 | 0.1063 |
| 2a turn only | 0.1837 | 0.1827 |
| 2a turn + history | 0.4539 | 0.4379 |

Rules:
- Open or proprietary models/APIs allowed; ALL must be disclosed with versions.
- Official retrieval must use ONLY the organizer corpus. No web retrieval, no external collection added to the index.
- Do not infer, hand-label, share, or reconstruct hidden gold judgments.
- Team limits, daily submission caps, hardware reporting, late policy: **[VERIFY]** when the platform is announced.

Dates:
- 30 Aug 2026: train/dev released
- 10 Jan 2027: evaluation window opens · 31 Jan 2027 (latest): closes
- Feb 2027: system papers (tentative) · Mar: notification · Apr: camera-ready **[VERIFY]**

---

## 5. Integrity rules (non-negotiable)

1. **One account, one team.** Never create or suggest a second account. Multi-account = disqualification risk.
2. **Dev is sacred.** Tune and select on train only. Every dev evaluation is logged in the ledger with a reason.
   Budget: at most one dev eval per major system version.
3. Never train, fine-tune, or build few-shot prompts using dev queries or dev qrels.
4. Never use gold_ids / gold_answers / guidance annotations at inference time. Train-split guidance may be used
   as a training signal only; state this explicitly in the paper.
5. Don't hard-code domain-specific hacks tuned to leaderboard feedback.
6. Zero fabricated numbers anywhere: code output, PROGRESS.md, slides, paper.
7. Record for every run: git commit, model names + versions/revisions, prompts, index config, seeds, API dates, runtime, hardware.

---

## 6. Environment & compute constraints

- Local: Windows + Git Bash, Dell Latitude i7 7th gen, 16 GB RAM, **no GPU**. Python, PyTorch, HF Transformers.
  Local = code, CPU smoke tests on tiny domains (IOTA: 10 queries, 10,372 docs), scoring, analysis.
- GPU: free Kaggle T4 (16 GB) / P100, ~30 GPU-hrs/week per account quota, ~20 GB working disk, ~30 GB RAM, 12 h session limit.
  Colab free as backup. Local LM Studio (Qwen2.5-7B-Instruct) is CPU-only and slow — not for bulk inference.
- Budget: assume **no paid APIs** unless Rafi explicitly approves a specific spend.
- Power/connectivity can drop: every long job must checkpoint and resume (per-domain caching, shard-wise embedding).
- Scripts must run on both Git Bash (Windows paths) and Kaggle Linux. Use `pathlib`, no hard-coded separators.

Compute planning rules:
- Estimate GPU-hours before proposing any run (docs × tokens × throughput). Say the estimate out loud.
- Embed corpora once per model, store fp16 per domain, reuse forever. 1.65M docs × 1024-d fp16 ≈ 3.3 GB.
- Truncate documents deliberately (check length distribution first); consider passage chunking only if data shows long docs hurt.
- Largest domain is History (801 queries, 356,493 docs) — it dominates query count but nDCG is macro over topics; check whether
  the official average is over topics globally or per-domain then macro **[VERIFY in scorer.py]**. This changes what to optimize.

---

## 7. Scope decision

- **Enter 1a + 1b with one pipeline.** 1b ranks per step; a 1a run is produced by fusing step-level rankings (+ whole-query retrieval).
- Track 2 is **out of scope** unless ALL hold by 1 Dec 2026: Track 1 system frozen, Task 6 on schedule, and Rafi approves.
  If added, only 2a (the retrieval stack transfers; history-aware query rewriting is the main lever).
- Rationale: largest headroom (BM25 ≈ 0.09), smallest engineering surface, direct reuse for Bangla Legal RAG.

---

## 8. Research agenda (read before building)

Phase-0 reading. For each: read the actual paper/repo, write a 5–10 line note in `notes/lit/<name>.md` with what they did,
numbers on TEMPO/BRIGHT if reported, and what we can reuse under our compute. Mark anything not confirmed as [UNVERIFIED].

Must-read:
1. **TEMPO** paper + repo (the source benchmark; their baselines, which retrievers/rerankers they tested, where models fail).
2. **RECOR** paper (only skim; for context).
3. **BRIGHT** (reasoning-intensive retrieval benchmark) — especially the finding that LLM-generated reasoning before retrieval helps.
4. Reasoning-oriented retrievers/rerankers evaluated on BRIGHT (e.g. ReasonIR, Rank1, ReasonRank, RaDeR) — check which have
   open weights that fit a T4.
5. Strong open embedders/rerankers: BGE-M3, bge-reranker-v2-m3, Qwen3-Embedding / Qwen3-Reranker (small sizes), E5 family.
   Check MTEB/BRIGHT numbers and licenses on Hugging Face.
6. Query expansion: HyDE, query2doc, LLM decomposition; Reciprocal Rank Fusion (Cormack et al. 2009).
7. Temporal IR / time-aware retrieval literature (temporal query intent, time expressions, event-period matching).
8. Anything the RETECO organizers (Abdallah, Jatowt et al.) have published on temporal or reasoning retrieval — organizers' own
   observations often point at the intended difficulty.

Deliverable of Phase 0: `notes/lit/SUMMARY.md` — a ranked list of 5–8 candidate techniques with expected gain, GPU cost on T4,
and risk. Then ONE recommended build order.

Hypotheses to test (each becomes an ablation row):
- H1: Dense/hybrid retrieval beats BM25 by a large margin on Track 1 (temporal ≠ lexical).
- H2: Using official decomposed steps (1b) and fusing step rankings improves 1a over whole-query retrieval.
- H3: LLM query rewriting that makes time constraints explicit (periods, before/after, trend) improves recall@100.
- H4: A cross-encoder / reasoning reranker over top-50–100 gives the biggest nDCG@10 jump per GPU-hour.
- H5: Gains differ by domain group (Blockchain / Social Sciences / Applied / STEM) — analyze, don't assume.
- H6: Learning from train-split `guidance_*.jsonl` (temporal annotations) helps a reranker or rewriter. [check what the field contains first]

---

## 9. Build plan with gates

Each phase ends with a gate. Don't start the next phase until the gate result is logged and Rafi has seen it.

**Phase 1 — Wiring (CPU, IOTA only)**
- Download only `track1_tempo/iota/*`. Run official baseline on train+dev.
- Gate: our numbers match `BASELINE_RESULTS.md` for IOTA exactly (to 4 decimals).

**Phase 2 — Full BM25 reproduction + data audit**
- Download all Track 1 domains. Reproduce official macro nDCG@10 for 1a and 1b (train).
- Audit: doc length distribution, query/step length, gold per query, duplicates, doc id formats, what `guidance` contains.
- Gate: reproduction matches official table; `notes/data_audit.md` written.

**Phase 3 — Validation harness**
- Build `eval/cv.py`: train-only evaluation with fixed query folds (e.g. 5-fold, stratified by domain), macro nDCG@10 +
  recall@100 + per-domain table, bootstrap 95% CI, paired comparison vs. previous best.
- Gate: harness reproduces BM25 numbers; variance across folds reported.

**Phase 4 — First-stage retrieval**
- Dense (one strong embedder that fits T4), then hybrid BM25+dense with RRF. Cache embeddings per domain.
- Measure recall@100 and recall@1000 (reranker ceiling) plus nDCG@10.
- Gate: pick first-stage config on train CV only.

**Phase 5 — Step fusion (1b → 1a)**
- Per-step retrieval; fusion variants (RRF, weighted by step, max/sum); include whole-query list.
- Gate: H2 answered with CI.

**Phase 6 — Query rewriting (optional, cost-gated)**
- Local LLM (quantized, fits T4) generates temporal rewrites / hypothetical docs. Cache all generations to disk.
- Gate: gain > CI width and GPU cost acceptable for the hidden test set size.

**Phase 7 — Reranking**
- Cross-encoder over top-K; then try a reasoning reranker if it fits. Tune K on train.
- Gate: H4 answered; latency/GPU-hours logged.

**Phase 8 — Freeze + single dev check (by ~20 Dec 2026)**
- Freeze v1. ONE dev evaluation. If dev ≪ train CV, investigate overfitting before changing anything.
- Build `submit/make_runs.py`: end-to-end from raw test files → validated TREC runs for 1a and 1b, with format checker.
- Dry-run on dev files as if they were test (no qrels read).

**Phase 9 — Evaluation window (10–31 Jan 2027)**
- Run frozen pipeline on test. Validate format. Submit early in the window; keep fallback runs (BM25+dense hybrid) ready.
- Respect submission caps; log every submission with commit hash.

**Phase 10 — Paper (Feb 2027)**
- Follow organizer checklist: entered sub-tracks + run ids, all model/API versions, query rewriting & temporal strategy,
  training data/preprocessing/indexing/hyperparameters, ablations by domain, compute + runtime, limitations & failure cases.
- Core contribution = the answered hypotheses (esp. where temporal grounding fails), not leaderboard rank.
- ACL/SemEval format, plain factual tone.

---

## 10. Repo layout

```
reteco/
  CLAUDE.md               # this file
  PROGRESS.md             # living status: phase, verified results, next step, blockers
  README.md               # how to reproduce everything
  configs/                # yaml per experiment
  reteco/                 # package: data.py, bm25.py, dense.py, fusion.py, rewrite.py, rerank.py, runs.py
  eval/                   # cv.py, score.py (wraps pytrec_eval), bootstrap.py
  kaggle/                 # notebooks/scripts to run on Kaggle, each resumable
  submit/                 # make_runs.py, format check wrapper
  notes/lit/              # literature notes + SUMMARY.md
  notes/data_audit.md
  results/ledger.csv      # one row per run (see below)
  results/runs/           # TREC run files (gitignored if large)
  cache/                  # embeddings, LLM generations (gitignored)
  paper/                  # LaTeX
```

`results/ledger.csv` columns:
`date, run_id, git_commit, phase, split(train_cv|dev|test), subtrack, config, models+revisions, macro_ndcg10, ci_low, ci_high,
recall100, gpu_hours, hardware, notes, dev_eval_reason`

---

## 11. Coding standards

- Python 3.10+, type hints, `argparse` CLIs, deterministic seeds, `tqdm` progress, logging to file.
- Every script: resumable (skip finished domains/shards), idempotent, prints where outputs went.
- Pin package versions in `requirements.txt`; pin HF model revisions (commit hashes) in configs.
- Unit-test the scorer wrapper, topic-id construction, and fusion on tiny synthetic data (`pytest`).
- Rafi's Claude Code skills live in `~/.claude/` (`/prd`, `/block`, `/test`, `/ship`) — use them when they fit.

---

## 12. Master TODO

- [ ] Create repo skeleton (§10), `PROGRESS.md`, empty ledger
- [ ] Phase 1: IOTA download + baseline match
- [ ] Join mailing list thread; ask organizers [VERIFY] items: test domains same as train/dev? averaging level? team/submission caps?
- [ ] Phase 0 literature notes + SUMMARY.md with one build order
- [ ] Phase 2: full Track 1 download, BM25 reproduction, data audit
- [ ] Phase 3: CV harness with CIs
- [ ] Phase 4: dense + hybrid first stage
- [ ] Phase 5: step fusion
- [ ] Phase 6: query rewriting (cost-gated)
- [ ] Phase 7: reranking
- [ ] Phase 8: freeze v1, single dev eval, end-to-end submission script, dry run
- [ ] Register team when platform opens (single account)
- [ ] Phase 9: submit in eval window
- [ ] Phase 10: system paper
- [ ] Export reusable modules + findings note for Bangla Legal RAG

---

## 13. What NOT to do

- Don't start Track 2 without the §7 conditions.
- Don't run full-corpus GPU jobs before a CPU smoke test on IOTA passes.
- Don't touch dev for model selection.
- Don't add external corpora to the index.
- Don't propose five things at once; propose the next one.
- Don't rewrite working code that wasn't part of the request.
- Don't write "state-of-the-art", "champion", or leaderboard comparisons with other teams in the paper.
