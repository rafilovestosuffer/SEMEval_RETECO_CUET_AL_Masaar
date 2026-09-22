# PROGRESS — SemEval-2027 Task 1 (RETECO) · CUET_AL_Masaar

> Living status file. Read this second, after `CLAUDE.md`, at the start of every session (§0).
> Update it at the end of every session: what changed, what is *verified*, what is next.
> A number appears here only if code in this repo produced it and it is in `results/ledger.csv` (§5.6).

Last updated: 2026-09-22

---

## Current phase

**Phase 2 — PASSED 2026-09-22.** Full Track 1 BM25 reproduction matches the organizers' published
table on **all 13 domains × 4 cells and all four macros**, to 4 dp. `notes/data_audit.md` written.
83 minutes of CPU, zero GPU quota spent to date across the whole project.

Phase 3 (validation harness on real data) has not started.

The `reteco/` core modules, the Phase 3 harness and the Phase 8 submission path are still **tested
in isolation only**. Phases 1–2 validated the *organizers'* code path, not ours. Note that
`restrict_to_corpus` is now known to be a **no-op** on this data (zero unreachable gold anywhere),
so it remains unexercised rather than validated — see the retraction below.

## Last verified result

**Phase 2 gate, all 13 domains, commit `6e2e1b7`** — ledger rows `phase2_bm25_full_r1`:

| | domain-macro (the gate) | query-macro (**the leaderboard**) | delta |
|---|---:|---:|---:|
| 1a train | 0.0879 ✓ | 0.0944 | +0.0066 |
| 1a dev | 0.0967 ✓ | 0.1055 | +0.0088 |
| 1b train | 0.0852 ✓ | 0.0910 | +0.0059 |
| 1b dev | 0.1063 ✓ | 0.1152 | +0.0088 |

Every per-domain cell matched too (52/52 counted). Verified by the kernel and independently by
`eval/gate.py --all` (exit 0). The left column reproduces the organizers' table; **the right
column is what ranks us** (§4) and is what the ledger records as `macro_ndcg10`.

## Next step (the ONE step)

**Phase 3 — validation harness on real data.** `eval/cv.py` and `eval/bootstrap.py` exist and are
unit-tested on synthetic fixtures only. Phase 2 supplies the real per-domain topic counts they
need, so the step is: produce per-topic BM25 scores for the train split, run 5-fold
domain-stratified CV, and confirm the harness reproduces the Phase 2 train numbers with fold
variance and bootstrap CIs reported.

Two changes to make first, both consequences of findings below:
1. The harness must report **both** aggregations, and treat the query-macro as primary.
2. Folds stay stratified by domain (that is right under either metric), but the objective they
   optimise is the query-macro, so History's 561 train queries dominate by design.

---

## Verified so far

| Date | What | Evidence |
|---|---|---|
| 2026-09-16 | Repo skeleton, `CLAUDE.md`, ledger, Kaggle control scripts committed | commit `71c9720` |
| 2026-09-16 | Kaggle / HuggingFace / RETECO docs site all unreachable from a Claude Code web session | 403 CONNECT; HF blocked even over the git proxy (`git ls-remote` on `tempo26/Tempo`) |
| 2026-09-16 | The RETECO **GitHub** repo *is* reachable via the git proxy | cloned at `23093c3`; starter kit read directly |
| 2026-09-16 | ~~Official metric is macro-averaged over the 13 domains~~ **SUPERSEDED 2026-09-22** — that is how the organizers aggregate their own *baseline table*; `evaluation.html` says the **leaderboard** macro-averages over *queries*. Both verified from primary sources; see the reopened blocker below | `BASELINE_RESULTS.md` + `official_baseline.py:259` vs `evaluation.html` |
| 2026-09-16 | The official stack (pyserini Lucene + gensim `LuceneBM25Model` + `pytrec_eval`) runs end to end under JDK 21 | `official_baseline.py` exit 0 on a synthetic fixture; its runs pass the organizers' `format_checker.py` (800 lines, 8 topics, 0 errors) |
| 2026-09-16 | `eval/gate.py` reads real `official_baseline.py` output and returns the right verdict | 48 tests pass; correct FAIL on the fixture |
| 2026-09-16 | **Phase 0 literature done** — TEMPO, RECOR, ReasonIR, DIVER read directly; arXiv ids for TEMPO (2601.09523) and RECOR (2601.05461) verified | `notes/lit/*.md` + `SUMMARY.md` |
| 2026-09-16 | H1 confirmed from the source paper: dense/reasoning retrieval is ~3× BM25 on TEMPO (10.8 → 22–32 macro nDCG@10) | TEMPO Table 3, `notes/lit/tempo.md` |
| 2026-09-16 | ~~H2 is untested in the literature~~ **FALSIFIED 2026-09-22** by a deep-research pass: sub-query rank fusion is published — MMLF (Findings of NAACL 2025) and ReDI (arXiv 2509.06544), which already ran the sum/max/RRF ablation. A stage-aware study (arXiv 2606.08577) further argues decomposition *harms* first-stage retrieval. H2 must be reframed | `reports/Temporal retrieval research gaps.md` §2 |

| 2026-09-16 | BRIGHT read directly (2407.12883v4, ICLR 2025); the 12.2-point CoT figure confirmed at source | `notes/lit/bright.md` |
| 2026-09-16 | **Correction to our own Phase 0 output**: MS MARCO cross-encoders *hurt* on reasoning retrieval (BM25 14.3 → 8.3 at k=100). `SUMMARY.md` had ranked this +4–7 at #4; now demoted to last and flagged presumed-harmful | BRIGHT Table 3 |
| 2026-09-16 | Core modules + Phase 3 harness + submission path built and unit-tested offline (120 tests) | this commit |
| 2026-09-16 | `submit/make_runs.py` runs end to end on the fixture and its output passes the organizers' real `format_checker.py`; our fallback checker agrees with it exactly | 2400 lines, 8 topics, 0 errors, both checkers |
| 2026-09-22 | **Phase 1 gate PASSED on real IOTA data** — 1a train 0.0199, 1a dev 0.2083, 1b dev 0.3289, all to 4 dp | kernel run 2 + `eval/gate.py` exit 0; ledger `phase1_bm25_iota_r2`, commit `fa495d3` |
| 2026-09-22 | HF dataset layout is exactly `track1_tempo/<domain>/*` as assumed; 201 files (track1 143, track2 55). `split_manifest.json` is **top-level**, not per-domain | kernel Stage A.2 file listing |
| 2026-09-22 | **Kaggle ships JDK 17; pyserini's Lucene jars need 21** (class file version 65.0 vs 61.0). Kernel now installs and explicitly selects 21 | run 1 `UnsupportedClassVersionError`; fixed in `fa495d3` |
| 2026-09-22 | IOTA is **7 train + 3 dev** queries (10 total, as §6 said) and 16 train / 8 dev steps | `num_topics` in the baseline output — closes the open question in `notes/lit/SUMMARY.md` |
| 2026-09-22 | ~~`restrict_to_corpus` semantics corroborated~~ **RETRACTED same day** — `num_topics=7` is just IOTA's train split size (7 train / 3 dev), not a drop. Phase 2 found **zero** unreachable gold ids in all of Track 1, so `restrict_to_corpus` is a no-op on this data and remains **unexercised** | `notes/data_audit.md` §4 |
| 2026-09-22 | **Phase 2 gate PASSED** — all 13 domains × 4 sub-track/split cells and all four macros match `BASELINE_RESULTS.md` @23093c3 to 4 dp | `eval/gate.py --all` exit 0; ledger `phase2_bm25_full_r1`; 83 min CPU, zero GPU |
| 2026-09-22 | Corpus is **1,654,055 docs / 4.44 GB**; splits match the official table exactly (1a 1211/519, 1b 2762/1214) | `cache/p2/data_audit.json` |
| 2026-09-22 | **29.4% of the corpus is byte-identical duplicate text** (485,683 docs; History 43.7%, bitcoin 49.9%); zero duplicate ids, zero empty docs | `notes/data_audit.md` §1 |
| 2026-09-22 | **H6 precondition met**: `guidance.query_guidance.temporal_reasoning_class_primary` exists — 13 classes. But TCP (`trends_changes_and_cross_period`) is only **9.7%** of train, and 5.0% of train queries are `is_temporal_query: false` | `notes/data_audit.md` §5 |
| 2026-09-22 | Measured gap between the two aggregations on BM25: query-macro exceeds domain-macro by +0.0059 to +0.0088 (7–9% relative) | `notes/data_audit.md` §7 |

Explicitly **not** verified: the Kaggle **GPU** path (the smoke kernel still has not been run);
any retrieval method of our own (Phases 1–2 ran the organizers' code, not ours); `eval/cv.py` and
`eval/bootstrap.py` against real data. The JDK, HF-layout and macro questions were settled by
Phases 1–2. Newly known: **HuggingFace is reachable from Rafi's laptop**, so small files (e.g. all
26 guidance files, 5.9 MB) can be pulled locally without spending a Kaggle run.

## Built but unverified against real data

Tested in isolation on a synthetic fixture. **No gate below is passed** — each needs the corpus.

| Module | What it does | What would falsify it |
|---|---|---|
| `reteco/data.py` | release-schema loaders, `restrict_to_corpus`, the 1b query template | real files whose field names differ from the starter kit's |
| `reteco/runs.py` | TREC read/write, corpus-order tie-break | a divergence from `official_baseline.py` on tied scores |
| `reteco/fusion.py` | RRF, weighted/max/sum, min-max interpolation | nothing — pure arithmetic; the *choice* among them is H2 |
| `eval/score.py` | two-level macro over `pytrec_eval` | disagreement with the organizers' own numbers in Phase 1/2 |
| `eval/bootstrap.py`, `eval/cv.py` | domain-stratified folds, CIs, paired test | fold counts on real per-domain query counts |
| `submit/make_runs.py`, `check_format.py` | end-to-end runs + validation | real test-split file naming |

The retriever inside `make_runs.py` is a deliberate token-overlap placeholder, **not a
baseline**. Phase 4 replaces it.

---

## Blockers and open questions

### [VERIFY] — ask the organizers (mailing list: semeval-2027-reteco, abdelrahman.abdallah@uibk.ac.at)

1. Do the hidden **test queries come from the same 13 domains / corpora** as train+dev, or are there
   unseen domains? This decides whether per-domain tuning is safe at all.
2. **Team size limits, daily submission caps, hardware-reporting requirements, late policy** — not yet
   announced; re-check when the evaluation platform opens.
3. **Paper dates** (Feb 2027 system papers / Mar notification / Apr camera-ready) are marked tentative
   on the official site.

**REOPENED 2026-09-22 — averaging level. The 16 Sept resolution was wrong in the direction that
matters.** It concluded "per-domain, then equal-weight macro over the 13 domains" and that
History's 801 queries are worth 1/13, exactly like IOTA's ~10. The evidence it rested on is real
but describes the organizers' *baseline table*, not the leaderboard:

| | says | source |
|---|---|---|
| Leaderboard | macro over **queries** | `evaluation.html`: 1a "computed independently for each query and macro-averaged"; per-domain "reported **diagnostically**" |
| Baseline table | macro over **domains** | `BASELINE_RESULTS.md` "Macro-averaged over domains"; `official_baseline.py:259` `sum(vals)/len(vals)`, `num_topics` excluded |

Both are verified from primary sources. They are different numbers for different artifacts, and
the one that ranks us is the query macro. **Consequence: History is ~46% of train+dev and ~66% of
train, not 7.7% — 6× the weight we assumed**, which inverts the old advice that a gain on a small
domain is worth as much as one on a large domain. Optimize the query macro; keep the domain macro
only for the Phase 1/2 reproduction gates; report both on every run.

Still worth asking the organizers to confirm, because the two official documents disagree in
plain language and the Phase 2 run will let us state the size of the gap exactly.

### Operational

- Kaggle API key was pasted into a chat transcript on 2026-09-16 and **must be rotated**
  (kaggle.com/settings → Account → API → Expire Token, then Create New Token). No key is stored in
  this repo and none should ever be. *Status 2026-09-22: a working key is present in
  `~/.kaggle/kaggle.json` and authenticates as `rafiurrahman01`, but whether it is the rotated one
  or still the exposed one is unknown from here — Rafi must confirm.*
- **`kaggle/pull_output.py` cannot print a kernel log on Windows.** The Kaggle library writes the
  log with the cp1252 default encoding and IOTA's tqdm bars contain `▉` (U+2589), so the pull dies
  with `UnicodeEncodeError` and leaves a 0-byte `.log`. Workaround: `export PYTHONUTF8=1` before
  pulling. Not yet fixed in the script.
- Semester finals run to 20 Sept 2026 and the NuNO paper is due 30 Sept 2026. RETECO is secondary
  (§2) — do not propose work that assumes full-time availability before October.

---

## Environment notes

Two machines, and they can do different things:

| | Rafi's laptop (Git Bash, Windows) | Claude Code web session | Kaggle kernel |
|---|---|---|---|
| Internet | full | **PyPI + GitHub only** | per-kernel toggle |
| GPU | none | none | T4 / P100 |
| Role | downloads, Kaggle CLI, CPU smoke tests | writes code, runs offline tests | bulk embedding / reranking |

Consequence: every step that touches HuggingFace, the RETECO site, or the Kaggle API is a **local**
step. Claude Code sessions write the code; Rafi runs it and pastes back real output (§3).

---

## Phase gates

- [x] **Phase 1** — IOTA BM25 matches `BASELINE_RESULTS.md` to 4 dp *(PASSED 2026-09-22, commit
      `fa495d3`; verified by the kernel and independently by `eval/gate.py`)*
- [x] **Phase 2** — full Track 1 BM25 reproduction + `notes/data_audit.md` *(PASSED 2026-09-22,
      commit `6e2e1b7`; 52/52 per-domain cells and all four macros match to 4 dp)*
- [ ] **Phase 3** — CV harness reproduces BM25, fold variance reported
- [ ] **Phase 4** — first-stage config picked on train CV only
- [ ] **Phase 5** — H2 (step fusion) answered with CI
- [ ] **Phase 6** — query rewriting gain > CI width, cost acceptable
- [ ] **Phase 7** — H4 (reranking) answered, GPU-hours logged
- [ ] **Phase 8** — v1 frozen, single dev eval, submission dry-run
- [ ] **Phase 9** — submitted in the evaluation window
- [ ] **Phase 10** — system paper

Side quest, not a gate: Kaggle GPU smoke kernel runs and reports `torch.cuda.is_available() == True`.
