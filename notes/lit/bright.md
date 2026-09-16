# BRIGHT — the reasoning-intensive retrieval benchmark everything else is measured on

**Su, Yen, Xia, Shi, Muennighoff, Wang, Liu, Shi, Siegel, Tang et al.** *BRIGHT: A Realistic and
Challenging Benchmark for Reasoning-Intensive Retrieval.* arXiv 2407.12883; **ICLR 2025**.
<https://brightbenchmark.github.io/>

**Status: not read directly.** Everything below is taken from the TEMPO, RECOR, ReasonIR and DIVER
papers, which were read directly and agree with each other. Marked **[SECONDARY]** — re-check against
the paper before citing any of it in our own write-up.

## Why it matters to us

TEMPO positions itself explicitly against BRIGHT: BRIGHT is reasoning-intensive but has **no temporal
grounding**, TEMPO adds it. Almost every model we might use reports BRIGHT numbers and not TEMPO
numbers, so BRIGHT is our proxy for model selection — with the caveat that it is a proxy.

## Facts consistently reported across the four papers [SECONDARY]

- 1,384 real-world queries, 12 domains (Stack Exchange, coding, theorem-based).
- **The headline gap:** models scoring 59.0 nDCG@10 on standard benchmarks score only **18.3** on
  BRIGHT (quoted identically in RECOR §2 and DIVER §2.3, referring to SFR-Embedding-Mistral).
  BM25 gets 14.5–14.8.
- **Chain-of-thought reasoning before retrieval improves results by up to 12.2 points** (RECOR §2).
  This is the finding CLAUDE.md §8.3 flags, and it is the origin of the whole reason-query line of work.
- Current leaderboard ladder, from DIVER Table 2: ReasonIR+QwenRerank 36.9 → ReasonIR+Rank-R1-32B
  38.8 → RaDeR+QwenRerank 39.2 → XRR2 40.3 → ReasonRank 40.8 → DIVER v1 41.6 →
  BGE-Reasoner 45.2 → DIVER v2 45.8 → **DIVER v3 46.8**.
- Its corpus has known quality problems — truncated sentences, excessive blank lines, structural artifacts
  from web scraping, concentrated in the seven Stack Exchange subdomains. DIVER's DChunk and the
  separate BRIGHT+ work (arXiv 2506.07116) both exist to clean it.

## The transfer caveat, stated plainly

BRIGHT rank order does **not** reproduce on TEMPO. DiVeR leads both, but below the top the orders
diverge: on TEMPO, E5 (30.4) and SFR (30.0) beat ReasonIR (27.2), while on BRIGHT ReasonIR (24.4)
comfortably beats E5 (17.8 area) and SFR (18.3). TEMPO's own domains are Stack Exchange too, so this is
not a domain-shift artifact — plausibly the temporal dimension rewards different behaviour.

**Consequence for model selection:** prefer TEMPO numbers where they exist (TEMPO Table 3 covers 12
models), and treat BRIGHT only as a tiebreaker for models TEMPO did not test. This is why the build
order in `SUMMARY.md` ranks on TEMPO first.

## To do

Read the paper directly before Phase 4 if we intend to cite the 12.2-point CoT figure or the 18.3 headline
in our system paper. Both are load-bearing for the H3 argument and neither has been verified at source.
