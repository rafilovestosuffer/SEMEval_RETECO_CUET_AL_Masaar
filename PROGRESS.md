# PROGRESS — SemEval-2027 Task 1 (RETECO) · CUET_AL_Masaar

> Living status file. Read this second, after `CLAUDE.md`, at the start of every session (§0).
> Update it at the end of every session: what changed, what is *verified*, what is next.
> A number appears here only if code in this repo produced it and it is in `results/ledger.csv` (§5.6).

Last updated: 2026-09-16

---

## Current phase

**Phase 0 — repo bootstrap.** No retrieval code exists yet. No data has been downloaded.
The repo skeleton (§10), this file, the ledger and the Kaggle GPU control path are in place;
nothing beyond that has been built.

## Last verified result

**None.** `results/ledger.csv` contains only its header row. The official BM25 baseline numbers
quoted in `CLAUDE.md` §4 are the *organizers'* published figures — they have not yet been
reproduced by this repo, and must not be cited as ours until Phase 1's gate passes.

## Next step (the ONE step)

**Phase 1 — wiring, CPU, IOTA only.** Download `track1_tempo/iota/*`, run the official BM25
baseline (gensim `LuceneBM25Model`, k1=0.9, b=0.4) on train and dev, and match
`BASELINE_RESULTS.md` for IOTA to 4 decimals.

Blocked on: Rafi downloading the IOTA domain locally. Claude Code web sessions cannot reach
`huggingface.co` or the RETECO site (see *Environment notes* below), so the download is a local step.

---

## Verified so far

| Date | What | Evidence |
|---|---|---|
| 2026-09-16 | Repo skeleton, `CLAUDE.md`, ledger, Kaggle control scripts committed | this commit |
| 2026-09-16 | Claude Code web container cannot reach Kaggle / HuggingFace / RETECO site | proxy returns 403 CONNECT for all three; PyPI + GitHub reachable |

Nothing else. Explicitly *not* verified: any nDCG number, any data statistic, the Kaggle GPU
path end-to-end (the smoke kernel has not been run yet).

---

## Blockers and open questions

### [VERIFY] — ask the organizers (mailing list: semeval-2027-reteco, abdelrahman.abdallah@uibk.ac.at)

1. Do the hidden **test queries come from the same 13 domains / corpora** as train+dev, or are there
   unseen domains? This decides whether per-domain tuning is safe at all.
2. **Averaging level of the official metric**: is macro nDCG@10 averaged over all topics globally, or
   per-domain and then macro-averaged across domains? History alone has 801 of the queries, so the two
   differ a lot and they change what we optimize. (§6 — check `scorer.py` in the starter kit first.)
3. **Team size limits, daily submission caps, hardware-reporting requirements, late policy** — not yet
   announced; re-check when the evaluation platform opens.
4. **Paper dates** (Feb 2027 system papers / Mar notification / Apr camera-ready) are marked tentative
   on the official site.

### Operational

- Kaggle API key was pasted into a chat transcript on 2026-09-16 and **must be rotated**
  (kaggle.com/settings → Account → API → Expire Token, then Create New Token). No key is stored in
  this repo and none should ever be.
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

- [ ] **Phase 1** — IOTA BM25 matches `BASELINE_RESULTS.md` to 4 dp
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
