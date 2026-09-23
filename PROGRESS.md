# PROGRESS — SemEval-2027 Task 1 (RETECO) · CUET_AL_Masaar

> Living status file. Read this second, after `CLAUDE.md`, at the start of every session (§0).
> Update it at the end of every session: what changed, what is *verified*, what is next.
> A number appears here only if code in this repo produced it and it is in `results/ledger.csv` (§5.6).

Last updated: 2026-09-24

---

## Current phase

**Phase 8 — PASSED 2026-09-24. Single dev check of v1: 1a 0.3665, 1b 0.3147** (BM25 dev 0.1055 /
0.1121). Every stage replicates on dev with the same sign and order as train — dense +0.1857,
fusion +0.0080 [+0.0011, +0.0158], rerank +0.0673 [+0.0488, +0.0868] — and dev is not below train,
so there is no overfitting signal. Final dev runs: 519/519 queries reranked, 0 parse failures,
every file VALID. Nothing is left to do before the January evaluation window except the
organizer questions below.

**Phase 8 — v1 FROZEN 2026-09-23 (git tag `v1-frozen`).** v1 =
Diver-Retriever-0.6B dense → step fusion (1a) → ReasonRank-7B over the fused top-30 (1a);
1b is the plain dense step run. The frozen pipeline (README "Producing submission runs") was
dry-run on dev through fusion and assembly: 519/519 queries and 1,214/1,214 steps covered, every
file VALID under the organizers' checker, no qrels read.

**Phase 7 — H4 CONFIRMED 2026-09-23: reranking +0.0508 nDCG@10**, on a seeded random 896 of 1,211
train queries (the time budget ended the session). The cost was 5× my estimate — see below.

**Phase 5c — 1b mirror of H2 is NEGATIVE:** the parent query's pool never helps 1b.

**Phase 5b — operator ablation DONE 2026-09-23: operator does not matter once pools are unioned; config unchanged.**

**Phase 5 — H2 ANSWERED 2026-09-23, with a mechanism.** Step fusion beats whole-query
retrieval by +0.0115 nDCG@10 (paired CI excludes zero), and the gain comes from **candidate
pool union rather than score blending** — see below. This is the paper's contribution.

**Phase 4 — first stage COMPLETE 2026-09-23. H1 confirmed on our own data.**
Dense retrieval beats BM25 by 2.7x, and the corpus is embedded and reusable.

**Phase 3 — PASSED 2026-09-22.** The validation harness reproduces BM25 on real data, reports
fold variance and bootstrap CIs under both aggregations, and its paired test is calibrated
(null straddles zero across three seeds; detects a gain from improving 0.5% of queries).

**Phase 2 — PASSED 2026-09-22.** Full Track 1 BM25 reproduction matches the organizers' published
table on **all 13 domains × 4 cells and all four macros**, to 4 dp. `notes/data_audit.md` written.
83 minutes of CPU, zero GPU quota spent to date across the whole project.

The `reteco/` modules, the Phase 3 harness and the submission path are now **validated on
real data**: Phases 3-7 ran our own code end to end, and the frozen pipeline was dry-run on
dev (fusion + assembly) with every file VALID under the organizers' checker.
`restrict_to_corpus` is a **no-op** on this data (zero unreachable gold), so it remains
unexercised rather than validated.

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

## Phase 4 result — H1 answered

`AQ-MedAI/Diver-Retriever-0.6B` against the organizers' BM25, train split, both scored by our
own harness (ledger `phase4c_dense_search`, commit `28ba526`):

| | dense | BM25 | paired delta |
|---|---:|---:|---|
| 1a nDCG@10 (query macro) | **0.2567** | 0.0944 | **+0.1622** [+0.1472, +0.1774] SIGNIFICANT |
| 1b nDCG@10 (official agg.) | **0.2792** | 0.0919 | **+0.1874** [+0.1711, +0.2040] SIGNIFICANT |
| 1a recall@100 | **0.648** | 0.268 | — |

Domain macros are 0.2201 (1a) and 0.2544 (1b), for comparison with the organizers' table.

**Three consequences that change the plan:**

1. **Reranking is back on the table.** Phase 3 measured BM25 recall@100 at 0.268 and 56.6% of
   queries with no gold in the top 100, which capped any reranker near 0.27 and was the main
   evidence for the research pass's "do not rerank" conclusion. Dense recall@100 is **0.648** —
   two and a half times the ceiling that argument rested on. The argument does not survive its
   premise; Phase 7 should be re-decided on this number, not on the BM25 one.
2. **Fusing BM25 is not worth it, now measured rather than predicted.** Top-100 overlap between
   the two arms is **0.060** — they agree on almost nothing — but the disagreement is nearly all
   one-directional: dense finds **1,706** judged documents BM25 misses, BM25 finds **104** dense
   misses. Union recall@100 is 0.675 against dense's 0.648, so a perfect candidate-pool merge
   buys at most **+0.027**, and score fusion would dilute a 2.7x-stronger arm to get it. Low
   overlap was the usual argument *for* fusion; the marginal-gold columns show why it is not
   sufficient on its own.
3. **The recall ceiling fear was unfounded.** The research pass put the realistic recall@100
   ceiling near 0.45, extrapolated from BRIGHT tables on corpora 4-200x smaller. We measure 0.648
   on 1.65M documents with a 0.6B model.

Per-domain, dense wins everywhere. The largest gains are on the domains BM25 handled worst —
history 0.0691 -> 0.2889 (and it is 46% of the metric), economics 0.0382 -> 0.1977, workplace
0.0777 -> 0.2920.

## Phase 5 result — H2, and why it works

Fusing each query's 1b Query+Step rankings with its 1a ranking, scored against the 1a qrels,
train split (ledger `phase5_stepfuse_*`):

| parent weight | query macro | vs whole-query | paired 95% CI |
|---|---:|---:|---|
| 0 (steps only) | 0.2584 | +0.0014 | [−0.0048, +0.0074] **ns** |
| **0.001** | **0.2685** | **+0.0115** | [+0.0071, +0.0162] **significant** |
| 0.25 | 0.2668 | +0.0098 | [+0.0060, +0.0137] significant |
| 5 | 0.2600 | +0.0030 | [+0.0010, +0.0050] significant |
| 10 | 0.2584 | +0.0014 | [−0.0003, +0.0030] ns |

Whole-query baseline 0.2570. The curve is flat from w=0.001 to ~0.25, then declines
monotonically as the parent dilutes the step signal.

**The mechanism is pool union, not score blending.** A parent weight of **0.1%** captures the
entire gain, while w=0 captures none — because w=0 removes the parent's documents from the
candidate pool rather than merely down-weighting them. Candidate-pool recall@100 makes it
explicit:

| pool | recall@100 |
|---|---:|
| whole-query only | 0.6479 |
| steps only | 0.6466 |
| **union** | **0.6849** |

The two retrieval formulations find **nearly the same amount** of gold and **different gold** —
each adds about +0.037 recall over the other. So the parent query's contribution is *coverage*,
not ranking signal, and the honest description of the method is a candidate-pool union with a
tie-break, not a weighted score fusion.

This matches what Phase 4 found for BM25 and dense, where pooling had headroom (+0.027 union
recall) while score fusion would have diluted the stronger arm. Two independent instances of
the same lesson: on this benchmark, **merge candidate pools, do not blend scores.**

## Phase 5b result — the operator does not matter, the weight does

Train split, 3 operators × 3 normalisations × 5 parent weights, all against the 1a qrels
(ledger `phase5b_*`, commit `38f4561`, local CPU only, dev untouched). The prediction from
the pool-union mechanism held:

| config | query macro | vs Phase 5 pick (sum/theoretical, w=0.001) |
|---|---:|---|
| max / rank | 0.2695 | +0.0010 [−0.0026, +0.0044] ns |
| sum / rank, w=0.001 | 0.2691 | +0.0006 [−0.0014, +0.0026] ns |
| sum / minmax, w=0.001 | 0.2690 | +0.0004 [−0.0001, +0.0010] ns |
| **sum / theoretical, w=0.001** | **0.2685** | — |
| RRF, w=0.001 | 0.2683 | −0.0003 [−0.0023, +0.0017] ns |
| max / theoretical | 0.2654 | −0.0032 [−0.0070, +0.0004] ns |
| **RRF, w=1 (textbook equal weight)** | 0.2625 | **−0.0060 [−0.0093, −0.0031] significant** |

- Every config with the parent in the pool lands in 0.2654–0.2695 (+0.008 to +0.013 over
  whole-query, all significant); every w=0 config is ns. **Pool membership decides the
  result, and the operator does not.**
- The one significant difference is textbook equal-weight RRF, and at w=0.001 RRF recovers
  fully. It loses because of the parent's **weight**, not because it is RRF, which matches
  the Phase 5 curve.
- `max` ignores `parent_weight` in `reteco/stepfuse.py` (unweighted max over lists), so it
  is flat across w. That makes it the purest "union + best-score" operator, and it is the
  top number here, but not significantly so.
- **Decision: keep sum / theoretical / w=0.001.** Nothing beat it significantly, and
  switching would pick a best-of from noise (§5.2 spirit). Per-domain, History moves
  0.2901 → 0.2987 (it is 561 of 1,211 train queries). law, monero and quant dip slightly, and
  those three are 94 queries combined.

## Phase 5c result — the parent pool does not help 1b

Each step's ranking fused with its parent query's 1a ranking, scored against the step qrels
with the official 1b aggregation (ledger `phase5c_1b_parentfuse_w1`, `eval/stepfuse_1b.py`):
w=0.001 changes nothing (a step's top-10 is already full; parent-only documents land below it),
and every larger weight **hurts**, monotonically: −0.0018 at w=0.1 to −0.0083 [−0.0120, −0.0047]
at w=1, all significant. **Pool union helps only in the 1a direction**, many step pools into
one query, and not the reverse. 1b stays the plain dense step run (0.2791).

## Phase 7 result — H4, reranking works, and it is expensive

Reranker chosen from the literature (`notes/lit/reasonrank.md`): on BRIGHT, reranking
ReasonIR's top-100 (30.59), most rerankers under 32B make the list *worse* (RankT5 16.60,
RankZephyr 22.64, Rank1-7B 27.23); only Rearank-7B (31.75) and **ReasonRank-7B (35.74)**
improve it. ReasonRank-7B (MIT, `liuwenhan/reasonrank-7B @3444046`), the authors' prompt and
settings ported verbatim into `reteco/rerank.py` (unit-tested), fused top-30 reranked with two
sliding windows, ranks 31–100 untouched. Ledger `phase7_rerank_reasonrank7b_d30`.

| on the same 896 random train queries | query macro |
|---|---:|
| dense (Phase 4) | 0.2670 |
| + step fusion (Phase 5) | 0.2787 |
| **+ ReasonRank-7B top-30** | **0.3296** |
| paired vs fused | **+0.0508 [+0.0362, +0.0664] significant** |

- 12 of 13 domains improve; bitcoin dips (0.1660 → 0.1538, 49 queries). History, 413 of these
  queries, goes 0.3142 → 0.3539. Wins 390, losses 194, ties 312.
- Parsing is not a risk: 2 unparseable windows of 1,792, 2 truncated reasoning chains.
- **The 896 are a seeded random sample, not the first 896**: the kernel shuffles before
  chunking so a budget cut leaves an unbiased subset. The other 315 train queries were not
  reranked; the full-train number is therefore not reported.
- **Cost, 5× over my estimate: 29.7 s/query on T4×2, 7.64 h wall.** Two errors: prompts are
  6.5k tokens per window, not ~3.8k (20 passages at up to 512 tokens each), and vLLM fell back
  to Triton attention because FlashAttention needs compute capability ≥ 8. Measured
  throughput ~490 tok/s total, prefill-bound. Consequences: dev (519 queries) ≈ 4.3 h; a test
  set above ~900 queries must be split across two sessions (the assembler already falls back
  to the fused list for any topic not reranked, so a split is safe).
- Whether T4×2 bills quota at 1× or 2× is **still unmeasured** and now matters: 7.64 h wall is
  7.6 or 15.3 GPU-hours.

## Next step (the ONE step)

**Phase 9, in the evaluation window (10–31 Jan 2027):** set `SPLIT = "test"` in the two pipeline
kernels and run the README's "Producing submission runs" sequence unchanged; submit only if
`submit/assemble_runs.py` exits 0. Before then, the only open items are the organizer questions
(test domains, submission caps, run naming) and registering the team when the platform opens.
Budget the rerank at ~33 s/query on T4×2 (dev: 519 queries in 4.75 h); above ~800 test queries,
split reranking across two sessions — the assembler falls back safely for anything not reached.

## Phase 8 result — the single dev check

| 1a, dev (519 queries) | query macro | paired step |
|---|---:|---|
| BM25 (official) | 0.1055 | — |
| dense | 0.2912 | +0.1857 [+0.1605, +0.2112] |
| + step fusion | 0.2992 | +0.0080 [+0.0011, +0.0158] |
| **+ ReasonRank-7B top-30 (v1)** | **0.3665** | +0.0673 [+0.0488, +0.0868] |

1b dev (v1 = plain dense steps): **0.3147** vs BM25 0.1121, +0.2026 [+0.1760, +0.2288].
Ledger `phase8_v1_dev`. Reranking helps 11 of 13 domains on dev; it lowers iota (3 queries) and
monero (19), and raises History, 240 of the 519 queries, from 0.3035 to 0.3810.

## Phase 4a notes (kept for the record)

- **Length-sorted batching is not a lever.** Measured 5,860 sorted vs 5,991 unsorted — nothing,
  because sentence-transformers already sorts internally. The earlier claim that padding cost
  ~3x *at runtime* was wrong. Padding is real in an **estimate** (1.65M x 512 assumes 0.85B
  tokens against 0.26B actual) but not in **execution**.
- **`torch.cuda.is_bf16_supported()` returned True on sm75** and must not be trusted; the T4 has
  no bf16 tensor cores. Set fp16 explicitly and verify `model.dtype`.
- **Kaggle allocated 2x T4**; only one was used. Whether a two-GPU session bills quota at 1x or
  2x is still unmeasured, and is worth ~2x wall-clock if it is 1x.


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

Newly known: **HuggingFace is reachable from Rafi's laptop**, so small files (query files,
guidance, qrels) are pulled locally without spending a Kaggle run. The Kaggle GPU path is
proven (Phases 4a-7).

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
  with `UnicodeEncodeError` and leaves a 0-byte `.log`. **Fixed**: `kaggle/_cli.py` runs the CLI
  with `PYTHONUTF8=1`. Separately, a full pull also downloads the kernel's `reteco_data/` and is
  slow; pull runs only with `kaggle kernels output ... --file-pattern "(runs/.*|.*\.log)"`.
  The same cp1252 crash hit the organizers' `format_checker.py`; fixed in `submit/check_format.py`.
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
- [x] **Phase 3** — CV harness reproduces BM25, fold variance reported *(PASSED 2026-09-22; both
      aggregations reported, paired test calibrated against a shuffle null)*
- [x] **Phase 4** — first-stage config picked on train CV only *(2026-09-23: Diver-Retriever-0.6B,
      1a 0.2567 vs BM25 0.0944, paired CI excludes zero; BM25 fusion rejected on measured evidence)*
- [x] **Phase 5** — H2 (step fusion) answered with CI *(2026-09-23: +0.0115 nDCG@10,
      [+0.0071,+0.0162]; mechanism is pool union, not score blending)*
- [ ] **Phase 6** — query rewriting gain > CI width, cost acceptable
- [x] **Phase 7** — H4 (reranking) answered, GPU-hours logged *(2026-09-23: ReasonRank-7B top-30 +0.0508 [+0.0362,+0.0664] on 896 random train queries; 7.64 h wall on T4x2)*
- [x] **Phase 8** — v1 frozen, single dev eval, submission dry-run *(2026-09-24: tag v1-frozen; dev 1a 0.3665, 1b 0.3147; full pipeline dry-run on dev, every file VALID)*
- [ ] **Phase 9** — submitted in the evaluation window
- [ ] **Phase 10** — system paper

Side quest, not a gate: Kaggle GPU smoke kernel runs and reports `torch.cuda.is_available() == True`.
