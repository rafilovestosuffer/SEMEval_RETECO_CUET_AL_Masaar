# PROGRESS — SemEval-2027 Task 1 (RETECO) · CUET_AL_Masaar

> Living status file. Read this second, after `CLAUDE.md`, at the start of every session (§0).
> Update it at the end of every session: what changed, what is *verified*, what is next.
> A number appears here only if code in this repo produced it and it is in `results/ledger.csv` (§5.6).

Last updated: 2026-09-16

---

## Current phase

**Phase 1 — wiring (CPU, IOTA only), built but not yet run.** The gate is coded and the
toolchain is proven; what is missing is one Kaggle run against the real IOTA domain.

## Last verified result

**No nDCG number exists yet.** `results/ledger.csv` still contains only its header row. The
BM25 figures in `CLAUDE.md` §4 and in `eval/gate.py` are the *organizers'* published values —
they have not been reproduced by this repo and must not be cited as ours until the gate passes.

## Next step (the ONE step)

**Run the Phase 1 kernel on Kaggle** (CPU, internet on, zero GPU quota):

```bash
python kaggle/push_kernel.py kaggle/kernels/phase1_bm25_iota
python kaggle/pull_output.py --kernel reteco-phase1-bm25-iota
```

It probes for a JDK and prints the real HF repo layout before downloading anything, so the run
is informative even if it stops early. Paste the log back. Gate = IOTA 1a train 0.0199,
1a dev 0.2083, 1b dev 0.3289, each to 4 dp. (1b train is published as 0.0000 and abstains —
matching it proves nothing, since a broken pipeline returns 0.0000 too.)

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

Explicitly **not** verified: any nDCG value on real data; the Kaggle GPU path (the smoke kernel
has still not been run); whether the Kaggle image ships a usable JDK; the internal layout of the
HF dataset repo. The last two are what the Phase 1 kernel's Stage A probes.

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

- [ ] **Phase 1** — IOTA BM25 matches `BASELINE_RESULTS.md` to 4 dp *(gate coded in `eval/gate.py`;
      kernel built and toolchain proven — awaiting one Kaggle run on real data)*
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
