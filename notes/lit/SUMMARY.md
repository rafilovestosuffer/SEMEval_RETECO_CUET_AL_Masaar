# Phase 0 deliverable — candidate techniques, ranked, with one build order

Written 16 Sept 2026 from papers read directly this session: TEMPO (2601.09523), RECOR (2601.05461),
ReasonIR (2504.20595), DIVER (2508.07995). BRIGHT is **[SECONDARY]** — see `bright.md`.

Ranking is on **expected macro nDCG@10 gain per T4-hour**, not on raw gain. Every estimate is an
estimate; nothing here is a measured result of ours, and no number below belongs in `results/ledger.csv`.

---

## The three facts that drive everything

1. **BM25 is 3× behind.** TEMPO Table 3: BM25 10.8 macro nDCG@10; dense 21–30; reasoning-aware
   27–32. RETECO's own baseline (0.0879 on our split) sits in the same place. The first-stage retriever is
   the whole game; nothing else on this list is close.
2. **MTEB/BEIR scores are worthless as a selection signal here.** SFR-Embedding-Mistral: 59.0 on BEIR,
   18.3 on BRIGHT. Off-the-shelf Qwen3-Embedding-4B: **5.6** on BRIGHT — worse than BM25 — while
   the *same backbone* fine-tuned by DIVER reaches 28.9. Only pick models with published TEMPO or
   BRIGHT numbers.
3. **A T4 is 16 GB and has no bf16** (compute capability 7.5; P100 is 6.0). That single constraint
   eliminates the 7–8B retrievers and the 32B rerankers, and it decides the ranking below.

---

## Ranked candidates

| # | Technique | Expected gain | T4 cost | Risk | Evidence |
|---|---|---|---|---|---|
| 1 | **DIVER-Retriever-4B as first stage** | **+15–20** over BM25 | ~6–10 GPU-h to embed 1.65M docs, once | Low | TEMPO 32.0 vs BM25 10.8; BRIGHT 28.9–31.9 |
| 2 | **Hybrid `0.5·dense + 0.5·BM25`**, min-max normalised | **+3** | ~0 (scores already computed) | Very low | DIVER 33.9→37.2; ReasonIR 29.9→32.0 |
| 3 | **Step fusion for 1a (H2)** | unknown — **our contribution** | ~0 extra retrieval (1b runs anyway) | Medium | *Untested in the literature* |
| 4 | **Cross-encoder rerank of top-100** | +4–7 | ~2–4 GPU-h per full pass | Medium | ReasonIR 29.9→36.9 (but with a 32B reranker) |
| 5 | **Temporal query rewriting (H3)** | +8 on reasoning retrievers, **−6 on BM25** | LLM call per query; cache everything | High (cost + hurts the sparse arm) | TEMPO Table 5 / Fig 8 |
| 6 | **Document cleaning + rechunk (DChunk)** | +0.5 dense, ~0 BM25 | one-off CPU | Low | DIVER Table 5 |
| 7 | **Smaller open embedders (BGE/SBERT/Inst-L)** | +11–14 over BM25 | ~1–2 GPU-h, <1B params | Very low | TEMPO Table 3 (22.0/24.9/24.8) |

**Not recommended:** ReasonIR-8B (16 GB fp16, does not fit, and DIVER-4B beats it at half the size);
`Diver-GroupRank-32B` and Qwen2.5-32B rerankers (far out of budget); Step-Only retrieval (a known
negative result — see below).

---

## The one recommended build order

**Phase 4 — first stage.** `AQ-MedAI/Diver-Retriever-4B-1020`, fp16, pinned revision. 4B ≈ 8 GB, fits a
T4 with headroom. Embed each domain once, store fp16, reuse forever (§6). Fall back to BGE-M3 (<1B) if
the 4B model will not fit alongside the batch. Then hybrid with BM25 at 0.5/0.5 on min-max normalised
scores, and compare against RRF. Select on **train CV only**, per-domain.

**Phase 5 — step fusion (H2), our actual contribution.** Two independent papers say decomposition used
*instead of* the query hurts: TEMPO Step-Only 14.6 ≪ Query+Step 26.4, and ReasonIR found LangChain
decomposition dropping Nomic 12.1→10.5 and GRIT-7B 20.4→17.3. **So never test Step-Only.** Test
fusion of *Query+Step* rankings (RRF, weighted, max/sum) plus the whole-query list, into a 1a ranking.
Nobody has published this; it is cheap because 1b retrieval runs anyway; and it is the finding the paper
should be built around.

**Phase 7 — reranking.** A small cross-encoder over top-100 (bge-reranker-v2-m3 class, <1B). The
published +7 comes from a 32B reranker, so budget +3–4 realistically. Measure recall@100 first — it is the
ceiling, and if it is already low the reranker cannot help.

**Phase 6 — rewriting, last and cost-gated.** Biggest single published gain (+8 to +13.7) but it is the most
expensive and the most dangerous: it *degrades* BM25 badly. If used, apply the rewrite to the dense arm
only and keep the original query for the sparse arm. Cache every generation to disk.

---

## Design constraints that fall out of the reading

- **Optimize per-domain.** The metric is an equal-weight macro over 13 domains, so Iota's ~10 queries
  weigh the same as History's 801. Per-domain nDCG on small domains is high-variance — Phase 3's folds
  must be stratified by domain with per-domain CIs.
- **Never rewrite the BM25 arm.** TEMPO Fig 8: BM25 falls 10.8 → 4.3–6.2 under LLM reasoning
  augmentation.
- **Do not build a time-first system.** Stripping temporal signals costs only 2.2 points; Temporal-Only
  queries collapse DiVeR from 32.0 to 17.7. Temporal reasoning needs topical grounding.
- **Check the encoder's context limit before generating long rewrites.** GRIT-7B was trained with a
  256-token query cap and cannot use long queries; ReasonIR scales to 2048. A long rewrite into a
  short-context encoder is wasted compute.
- **Sparse and dense genuinely disagree** — ReasonIR and BM25 share only 28.2% of their top-100. That
  disagreement is why fusion works, and it is worth measuring on our own data.

## Open questions to settle with data, not reading

- Does `guidance_*.jsonl` carry TEMPO's **reasoning class** labels (TCP, HAC, CAU…)? If so, a
  per-reasoning-class breakdown is a stronger paper axis than per-domain — TCP is the hardest class
  (DiVeR 23.9) and TCP is exactly what RETECO is about. Answers H6. **Check in Phase 2's audit.**
- Document length distribution — decides whether DChunk-style rechunking is worth anything here.
- Recall@100 and recall@1000 of the hybrid first stage — the reranker's ceiling.
- CLAUDE.md §6 states IOTA has 10 queries; the Phase 1 kernel prints `num_topics` and will confirm or
  correct it.
