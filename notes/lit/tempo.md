# TEMPO — the source benchmark for Track 1

**Abdallah, Ali, Abdul-Mageed, Jatowt.** *TEMPO: A Realistic Multi-Domain Benchmark for Temporal
Reasoning-Intensive Retrieval.* arXiv **2601.09523v1** [cs.IR], 14 Jan 2026. University of Innsbruck + UBC.
Repo <https://github.com/tempo-bench/Tempo>. Read directly 16 Sept 2026 (arXiv id now **verified**).

Same group as the RETECO organizers — Abdallah and Jatowt are authors on both. Their observations
point at the intended difficulty, so this is the highest-value read in §8.

## What it is

1,730 complex queries over 13 Stack Exchange domains requiring temporal reasoning (tracking change,
trends, cross-period comparison); 3,976 decomposed steps with gold docs per step; novel temporal
metrics (TP@k, TR@k, TC@k, NDCG|FC@k). RETECO Track 1 = this benchmark, re-split 70/30.

## The numbers that matter (Table 3, nDCG@10, macro over 13 domains)

| BM25 | BGE | Contriever | SBERT | Inst-L | Qwen | Rader | GritLM | ReasonIR | SFR | E5 | **DiVeR** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10.8 | 22.0 | 21.4 | 24.9 | 24.8 | 22.8 | 24.2 | 27.2 | 27.2 | 30.0 | 30.4 | **32.0** |

Best system reaches only 32.0 nDCG@10 and 71.4% Temporal Coverage@10. The benchmark is *far* from solved.

Per-domain spread is enormous: Politics 47.9, Iota 41.7, Law 40.8 vs Bitcoin 17.6, Monero 23.7, Quant 27.8.

## Answers to our hypotheses

- **H1 — dense ≫ BM25: CONFIRMED, decisively.** 10.8 → 22–30 (dense) → 27–32 (reasoning retrievers).
  Roughly **3× BM25**. This is the single largest lever available and it is not close.
- **H2 — step fusion: NOT TESTED by TEMPO, and our framing differs.** They compare three ways of
  *building the 1b query*: Step-Only 14.6, **Query+Step 26.4**, Query+All 25.9. Step-Only is far worse —
  isolated steps lack context. But they never fuse step-level *rankings* back into a 1a ranking, which is
  what H2 proposes. **This is a genuine open gap and our most likely real contribution.**
  (RETECO's official 1b already uses Query+Step, i.e. their best variant.)
- **H3 — temporal rewriting: model-dependent, and the headline number does NOT generalise.**
  Table 5 "Normalized" (explicit temporal intent tags) gives ReasonIR **+8.0** (27.3→35.3) — but
  **read the whole column, not that one cell** (completed 22 Sept 2026; the earlier version of this
  note recorded only ReasonIR and BM25 and was quietly misleading by omission):

  | effect of intent tags | value |
  |---|---|
  | ReasonIR | **+8.0** |
  | Contriever / Qwen / BGE / BM25 | +1.0 / +0.5 / +0.1 / +0.1 |
  | Inst-L / GritLM / SFR | −0.1 / −0.1 / −0.5 |
  | E5 / SBERT / **DiVeR** / Rader | −1.3 / −1.4 / **−1.8** / −2.6 |

  Eleven of twelve retrievers average about **−0.6**. ReasonIR is instruction-conditioned, which is
  most likely why only it responds to an appended instruction-like clause. **For a DIVER-family
  first stage, Table 5 predicts this treatment hurts.** The tags are also gold-derived (TEMPO's
  GPT-4o annotation layer) with no template, example or construction code published, so +8.0 is an
  **oracle ceiling on one model**, not an achievable gain.

  Figure 8: LLM-generated reasoning gives ReasonIR **+13.7** with GPT-4o (27.2→41.0) but **degrades
  BM25 to 4.3–6.2** — reasoning text dilutes lexical signal. DiVeR is flat (30.7–32.2), having
  internalised the reasoning.
  **Rule that survives: never send one augmented query to both arms of a hybrid** — raw text to the
  sparse arm, augmented text to the dense arm only. That is a config choice worth ~5 nDCG@10 of
  avoided loss at zero cost, and it is the durable part of H3.
- **H5 — domain variation: CONFIRMED.** See spread above. Also: no single model wins everywhere.

## Two findings that reframe the problem

- **Temporal signal matters less than the name suggests.** Stripping temporal signals costs only 2.2 points
  on average. But *Temporal-Only* queries collapse (DiVeR 32.0→17.7). Temporal reasoning needs topical
  grounding; time alone retrieves nothing. Do not build a time-first system.
- **Reasoning class drives difficulty more than domain.** Trends & Cross-Period Comparison (TCP) is
  hardest — even DiVeR only 23.9. Historical Attribution & Context (HAC) is easiest (SFR 51.8). A
  per-reasoning-class breakdown is a better paper axis than a per-domain one, if RETECO ships the labels
  (**[UNVERIFIED]** — check whether `guidance_*.jsonl` carries the reasoning class; §8 H6).

## RAG result worth knowing

No-retrieval (77.3) beats **every** retrieval configuration; oracle gold docs reach 80.5, BM25 drags it to
73.8. Their hypothesis: temporally incomplete retrieved evidence actively misleads the generator. Relevant
if Track 2 is ever entered, and a good limitation to cite.

## Reusable for us

- Their query construction is exactly what RETECO's `official_baseline.py` implements — already matched.
- Their per-domain table gives a sanity band for our own numbers: any dense retriever we build that lands
  far from 22–32 macro on full Track 1 has a bug, not a finding.
- Limitations they state: English only, Stack Exchange only, recency-skewed temporal distribution,
  LLM-assisted annotation may contain errors.
