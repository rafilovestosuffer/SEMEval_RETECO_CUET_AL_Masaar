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

Metric: **nDCG@10**, `pytrec_eval` `ndcg_cut_10`. **Two different aggregations exist and they are not the same number —
reopened 22 Sept 2026.**

1. **Leaderboard (what ranks us): macro over QUERIES.** `evaluation.html`, verbatim: 1a — "nDCG@10 is computed
   independently for each query and macro-averaged"; 1b — "step-specific nDCG@10 values are first averaged over the
   supplied steps for each query, then aggregated across queries". It adds that "per-domain results are also reported
   **diagnostically**", which says domains are not the ranking unit.
2. **The starter-kit baseline table: macro over DOMAINS.** `BASELINE_RESULTS.md` says "Macro-averaged over domains, as
   both papers do", and `official_baseline.py:259` does exactly that — `sum(vals)/len(vals)` over per-domain entries,
   with `num_topics` explicitly excluded from the mean. This is why the 13 published per-domain 1a-train values average
   to 0.08785 → the published 0.0879.

Both readings are evidenced; they describe different artifacts. The 16 Sept entry concluded (1) was (2) and was wrong
about which one ranks us. Consequences: **optimize for the query-macro**, keep the domain-macro only to reproduce the
organizers' table (that is what the Phase 1/2 gates check), and **report both on every run** — it is free. Ask the
organizers to confirm (`notes/organizer_questions.md`). The pure-Python `scorer.py` is the *approximate* zero-install path and also adds temporal
precision/coverage diagnostics (not ranking); the official number comes from `official_baseline.py` + `pytrec_eval`.

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
- GPU: free Kaggle T4 (16 GB) / P100, ~30 GPU-hrs/week per account quota, ~20 GB working disk, ~30 GB RAM.
  **GPU sessions cap at 9 h, not 12** — the 12 h limit applies to CPU-only sessions (corrected 22 Sept 2026).
  Kaggle also offers **T4×2** (two independent 16 GB T4s); whether that bills quota at 1× or 2× is
  undocumented and is worth measuring before relying on it.
  Colab free as backup. Local LM Studio (Qwen2.5-7B-Instruct) is CPU-only and slow — not for bulk inference.
- Budget: assume **no paid APIs** unless Rafi explicitly approves a specific spend.
- Power/connectivity can drop: every long job must checkpoint and resume (per-domain caching, shard-wise embedding).
- Scripts must run on both Git Bash (Windows paths) and Kaggle Linux. Use `pathlib`, no hard-coded separators.

Compute planning rules:
- Estimate GPU-hours before proposing any run (docs × tokens × throughput). Say the estimate out loud.
- **MEASURED on a Kaggle T4, 22 Sept 2026** (`kaggle/kernels/phase4a_embed_bench`, ledger
  `phase4a_embed_bench`). Diver-Retriever-0.6B, fp16, batch 8, max_len 512, 20k stratified documents:
  **6,134 real tokens/sec = 25 docs/sec**. The corpus is 1,654,055 docs / 0.81B tokens untruncated,
  falling to ~0.26B after a 512 cap and removing the 29.4% duplicates, so:

  | model | BRIGHT | hours for 0.26B | weeks of quota |
  |---|---:|---:|---:|
  | **Diver-Retriever-0.6B** | 25.2 | **11.8 (measured)** | 0.39 |
  | Diver-Retriever-1.7B | 27.3 | 33.4 (derived) | 1.11 |
  | Diver-Retriever-4B-1020 | 31.9 | 78.5 (derived) | 2.62 |

  Only the 0.6B row is measured; the others scale it by parameter count. **The project is comfortably
  feasible** — one and a half 9-hour sessions for the whole corpus.
- **fp32 → fp16 is the one big runtime lever, and it is real**: measured 1,847 → 5,991 tok/s, a **3.2×**
  speedup, with peak VRAM halved. HuggingFace defaults to fp32, so this must be set explicitly.
- **Length-sorted batching gave no measurable win — because sentence-transformers already sorts
  internally.** Measured 5,860 sorted vs 5,991 "unsorted", i.e. nothing. This corrects the earlier claim
  that padding cost ~3× *at runtime*. The padding factor is real in an **estimate** (multiplying
  1.65M × 512 assumes 0.85B tokens against 0.26B actual) but not in **execution** with
  sentence-transformers. Count real tokens when estimating; do not expect a speedup from sorting.
- **The T4 is compute-bound, not bandwidth-bound**, for batched encoding. Consequence: shortening
  `max_length` barely helps speed *per token*; the win from a short cap is simply fewer real tokens.
- **Smaller batches are faster here**: 6,073 tok/s at batch 8 against 5,350 at batch 128, and peak VRAM
  at batch 8 was **1.74 GB of 15.6**. The T4 is nowhere near memory-bound at this model size, so VRAM is
  not the binding constraint — quota is.
- **`torch.cuda.is_bf16_supported()` returns True on sm75 and must not be trusted.** The T4 has no bf16
  tensor cores; the flag reflects emulation. Set fp16 explicitly and verify `model.dtype` after loading.
  The fp16 cast is numerically safe here — measured 0 inf, 0 nan, norms within 5e-4 of 1.0.
- **Kaggle allocated 2× T4** for this session. Only one was used. Using both could roughly halve
  wall-clock, but whether it bills quota at 1× or 2× is still unmeasured.
- **Query-side work is ~1000× cheaper than corpus-side work** — 1,730 queries vs 1.65M documents. The
  conclusion stands; **the number originally cited for it does not (corrected 22 Sept 2026).** This bullet
  previously justified the 0.6B choice partly on TEMPO Table 5's "+8.0 from temporal-intent tagging".
  Reading the whole column rather than the ReasonIR cell: eleven of twelve retrievers average **−0.6**,
  and **DiVeR specifically is −1.8**, so that treatment is predicted to *hurt* our first stage. The tags
  are gold-derived with no published construction, making +8.0 an oracle ceiling on one
  instruction-conditioned model (`notes/lit/tempo.md`).

  What replaces it, and is measured on TEMPO itself: **equal-weight fusion of offline-generated
  reasoning views, 0.265 → 0.284 (+1.9) with no learning at all** (arXiv 2608.08940). A learned gate
  adds only ~+0.005 at matched K, and gated K=3 beat K=5 — so prune views, do not gate them. Generation
  is offline and the LLM is never called at retrieval time, which keeps it legal at inference.
- Embed corpora once per model, store fp16 per domain, reuse forever. Verified `config.json` values
  (22 Sept 2026), so the cache size follows from the model choice:

  | model | dim | cache, all 1.65M docs | deduplicated (70.6%) |
  |---|---:|---:|---:|
  | **Diver-Retriever-0.6B** (the Phase 4 pick) | 1024 | 3.39 GB | **2.39 GB** |
  | Diver-Retriever-1.7B | 2048 | 6.78 GB | 4.78 GB |
  | Diver-Retriever-4B-1020 | 2560 | 8.47 GB | 5.98 GB |

  The 0.6B lands back near this file's original 3.3 GB estimate; the 4B would have been 2.5× that and
  awkward beside a 4.44 GB corpus on a ~20 GB disk. Write each domain out as it is produced regardless.
  **None of the T4/P100 pair supports bf16** — set fp16 explicitly and check `model.dtype` after
  loading, because HuggingFace defaults to fp32 and some DIVER cards specify bf16
  (`notes/lit/diver.md`).
- Truncate documents deliberately (check length distribution first); consider passage chunking only if data shows long docs hurt.
- Largest domain is History (801 queries, 356,493 docs). **Corrected 22 Sept 2026 — this reverses the 16 Sept entry.**
  The leaderboard macro is over *queries* (§4), so History is worth ~46% of the train+dev score and ~66% of train
  alone, not 1/13. That is **6× the weight** the previous entry assumed, and it inverts the advice that followed from
  it: a gain on History is worth roughly sixty times the same gain on IOTA, not the same. Consequences:
  - Optimize for the query-macro. A change that helps IOTA and hurts History is almost certainly a net loss.
  - Still **report per-domain**, because the organizers report it diagnostically and because a per-domain table is what
    makes a failure analysis publishable — but do not optimize the equal-weight mean of it.
  - Phase 3's folds stay stratified by domain with per-domain CIs: stratification keeps every domain represented in
    every fold, which is right under either metric. Only the objective changes, not the fold design.
  - The small domains are high-variance *and* low-weight, so a headline gain driven by IOTA's ~10 queries is noise
    wearing a result's clothes. Check the per-domain table before believing any macro movement.

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
