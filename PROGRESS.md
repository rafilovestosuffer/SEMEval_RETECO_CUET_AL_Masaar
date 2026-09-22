# PROGRESS — SemEval-2027 Task 1 (RETECO) · CUET_AL_Masaar

> Living status file. Read this second, after `CLAUDE.md`, at the start of every session (§0).
> Update it at the end of every session: what changed, what is *verified*, what is next.
> A number appears here only if code in this repo produced it and it is in `results/ledger.csv` (§5.6).

Last updated: 2026-09-22

---

## Current phase

**Phase 1 — PASSED 2026-09-22.** The IOTA BM25 reproduction matches the organizers' published
figures to 4 dp on all three load-bearing rows. The toolchain is now proven end to end on real
data: HF download → starter kit at the pinned commit → `official_baseline.py` → our gate.

Phase 2 (full Track 1 reproduction + data audit) has not started.

The `reteco/` core modules, the Phase 3 harness and the Phase 8 submission path remain **tested
in isolation only** — see *Built but unverified* below. Phase 1 validated the organizers' code
path, not ours; the one piece of ours it did corroborate is `restrict_to_corpus` (see below).

## Last verified result

**Phase 1 gate, IOTA, commit `fa495d3`** — ledger rows `phase1_bm25_iota_r2`:

| key | published | observed | topics | counts |
|---|---|---|---|---|
| 1a train | 0.0199 | **0.0199** | 7 | yes |
| 1a dev | 0.2083 | **0.2083** | 3 | yes |
| 1b dev | 0.3289 | **0.3289** | 8 | yes |
| 1b train | 0.0000 | 0.0000 | 16 | abstains |

Verified twice: by the kernel and independently by `eval/gate.py` (exit 0). 22.4 s on a Kaggle
CPU kernel, **zero GPU quota**. These are single-domain IOTA numbers, not the 13-domain macro,
and 1a dev is 3 topics — wiring evidence, not performance evidence. Do not quote them as a
system result.

## Next step (the ONE step)

**Phase 2 — full Track 1 BM25 reproduction + data audit.** Same kernel shape, all 13 domains
instead of one; gate is the published macro (1a train 0.0879, 1a dev 0.0967, 1b train 0.0852,
1b dev 0.1063) and `notes/data_audit.md`.

Estimate before proposing the run: IOTA is 10,372 docs and indexed in ~22 s, so 1.65 M docs is
roughly 1 h of CPU indexing, plus download. Still zero GPU. The audit questions are already
listed in `notes/lit/SUMMARY.md` — above all **whether `guidance_*.jsonl` carries TEMPO's
reasoning-class labels (TCP, HAC, CAU…)**, which would make a per-reasoning-class breakdown a
stronger paper axis than per-domain, and answers H6.

---

## Verified so far

| Date | What | Evidence |
|---|---|---|
| 2026-09-16 | Repo skeleton, `CLAUDE.md`, ledger, Kaggle control scripts committed | commit `71c9720` |
| 2026-09-16 | Kaggle / HuggingFace / RETECO docs site all unreachable from a Claude Code web session | 403 CONNECT; HF blocked even over the git proxy (`git ls-remote` on `tempo26/Tempo`) |
| 2026-09-16 | The RETECO **GitHub** repo *is* reachable via the git proxy | cloned at `23093c3`; starter kit read directly |
| 2026-09-16 | **Official metric is macro-averaged over the 13 domains, not over topics** | `BASELINE_RESULTS.md` prose + `official_baseline.py` aggregation; the 13 per-domain 1a-train values average to 0.08785 → published 0.0879. CLAUDE.md §4/§6 corrected |
| 2026-09-16 | The official stack (pyserini Lucene + gensim `LuceneBM25Model` + `pytrec_eval`) runs end to end under JDK 21 | `official_baseline.py` exit 0 on a synthetic fixture; its runs pass the organizers' `format_checker.py` (800 lines, 8 topics, 0 errors) |
| 2026-09-16 | `eval/gate.py` reads real `official_baseline.py` output and returns the right verdict | 48 tests pass; correct FAIL on the fixture |
| 2026-09-16 | **Phase 0 literature done** — TEMPO, RECOR, ReasonIR, DIVER read directly; arXiv ids for TEMPO (2601.09523) and RECOR (2601.05461) verified | `notes/lit/*.md` + `SUMMARY.md` |
| 2026-09-16 | H1 confirmed from the source paper: dense/reasoning retrieval is ~3× BM25 on TEMPO (10.8 → 22–32 macro nDCG@10) | TEMPO Table 3, `notes/lit/tempo.md` |
| 2026-09-16 | H2 is **untested in the literature** — TEMPO compares 1b query *constructions*, never fuses step rankings into 1a | TEMPO Fig 6, `notes/lit/SUMMARY.md` |

| 2026-09-16 | BRIGHT read directly (2407.12883v4, ICLR 2025); the 12.2-point CoT figure confirmed at source | `notes/lit/bright.md` |
| 2026-09-16 | **Correction to our own Phase 0 output**: MS MARCO cross-encoders *hurt* on reasoning retrieval (BM25 14.3 → 8.3 at k=100). `SUMMARY.md` had ranked this +4–7 at #4; now demoted to last and flagged presumed-harmful | BRIGHT Table 3 |
| 2026-09-16 | Core modules + Phase 3 harness + submission path built and unit-tested offline (120 tests) | this commit |
| 2026-09-16 | `submit/make_runs.py` runs end to end on the fixture and its output passes the organizers' real `format_checker.py`; our fallback checker agrees with it exactly | 2400 lines, 8 topics, 0 errors, both checkers |
| 2026-09-22 | **Phase 1 gate PASSED on real IOTA data** — 1a train 0.0199, 1a dev 0.2083, 1b dev 0.3289, all to 4 dp | kernel run 2 + `eval/gate.py` exit 0; ledger `phase1_bm25_iota_r2`, commit `fa495d3` |
| 2026-09-22 | HF dataset layout is exactly `track1_tempo/<domain>/*` as assumed; 201 files (track1 143, track2 55). `split_manifest.json` is **top-level**, not per-domain | kernel Stage A.2 file listing |
| 2026-09-22 | **Kaggle ships JDK 17; pyserini's Lucene jars need 21** (class file version 65.0 vs 61.0). Kernel now installs and explicitly selects 21 | run 1 `UnsupportedClassVersionError`; fixed in `fa495d3` |
| 2026-09-22 | IOTA is **7 train + 3 dev** queries (10 total, as §6 said) and 16 train / 8 dev steps | `num_topics` in the baseline output — closes the open question in `notes/lit/SUMMARY.md` |
| 2026-09-22 | `restrict_to_corpus` semantics corroborated against the organizers' own output: 1a train scores over 7 topics, not 10 | `num_topics=7`; our `reteco/data.py` drops the same topics |

Explicitly **not** verified: any nDCG value beyond IOTA; the 13-domain macro; the Kaggle **GPU**
path (the smoke kernel still has not been run). The JDK question and the HF layout question were
both settled by Phase 1 — see the table above.

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

**Resolved 2026-09-16 — averaging level.** Was: "is macro nDCG@10 over all topics globally, or
per-domain then macro?" Answer: **per-domain, then equal-weight macro over the 13 domains**
(evidence in the table above and in `notes/starter_kit_findings.md` §1). No need to ask.
Consequence: History's 801 queries are worth 1/13, exactly like IOTA's ~10 — optimize per-domain,
and expect high variance on the small domains.

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
- [ ] **Phase 2** — full Track 1 BM25 reproduction + `notes/data_audit.md`
- [ ] **Phase 3** — CV harness reproduces BM25, fold variance reported
- [ ] **Phase 4** — first-stage config picked on train CV only
- [ ] **Phase 5** — H2 (step fusion) answered with CI
- [ ] **Phase 6** — query rewriting gain > CI width, cost acceptable
- [ ] **Phase 7** — H4 (reranking) answered, GPU-hours logged
- [ ] **Phase 8** — v1 frozen, single dev eval, submission dry-run
- [ ] **Phase 9** — submitted in the evaluation window
- [ ] **Phase 10** — system paper

Side quest, not a gate: Kaggle GPU smoke kernel runs and reports `torch.cuda.is_available() == True`.
