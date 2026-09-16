# BRIGHT — the reasoning-intensive retrieval benchmark everything else is measured on

**Su, Yen, Xia, Shi, Muennighoff, Wang, Liu, Shi, Siegel, Tang, Sun, Yoon, Arık, Chen, Yu**
(HKU / Princeton / Stanford / UW / Google Cloud AI). *BRIGHT: A Realistic and Challenging Benchmark
for Reasoning-Intensive Retrieval.* arXiv **2407.12883v4**, **ICLR 2025**.
<https://github.com/xlang-ai/BRIGHT> · <https://huggingface.co/datasets/xlangai/BRIGHT>

**Read directly 16 Sept 2026.** The earlier `[SECONDARY]` version of this note was written from what
the TEMPO/RECOR/ReasonIR/DIVER papers said about BRIGHT; those marks are now cleared, and one claim
in it was wrong. See §4 below — it changes our Phase 7 plan.

## What it is

1,384 real-world queries, 12 datasets, three groups: seven Stack Exchange domains (relevance =
document cited in the accepted answer, unanimously confirmed by multiple annotators), two coding
(LeetCode, Pony), three theorem-based (AoPS, TheoremQA question + theorem retrieval). Framed as
"level 3" retrieval: not keyword (level 1), not semantic (level 2), but relevance that requires
deliberate reasoning to establish.

TEMPO positions itself directly against this: BRIGHT is reasoning-intensive but has **no temporal
grounding**; TEMPO adds it. Most models we might use report BRIGHT numbers and not TEMPO numbers,
so BRIGHT is our proxy for model selection — with the transfer caveat in §5.

## 1. The headline gap — and an inconsistency worth recording

> "The leading model on the MTEB leaderboard, SFR-Embedding-Mistral, which achieves a score of
> **59.0** nDCG@10 [on BEIR], produces a score of nDCG@10 of **18.3** on BRIGHT."

That 59.0 → 18.3 collapse is the number everyone quotes, and it is about SFR specifically.

**Inconsistency in the paper itself:** the abstract, Table 2's caption and the conclusion all say the
best model reaches **24.3** average nDCG@10, but Table 2's actual maximum is **Qwen (gte-Qwen1.5-7B)
at 22.5**. Probably a text/table mismatch across arXiv revisions (v4). **Cite neither as settled**; if
we need the figure in our paper, quote Table 2 and the model name, not the abstract.

Baselines from Table 2 (nDCG@10, avg over 12): BM25 **14.5**; BGE 13.7, Inst-L 14.2, SBERT 14.9
(all <1B, i.e. BM25 matches them); E5 17.9, SFR 18.3, Inst-XL 18.9, GritLM 21.0, **Qwen 22.5**;
proprietary Cohere 16.6, OpenAI 17.9, Voyage 17.9, Google 20.0.

## 2. The 12.2-point CoT figure — CONFIRMED at source

> "We show that incorporating explicit reasoning about the query improves retrieval performance by
> **up to 12.2 points**."

This was the load-bearing `[SECONDARY]` claim behind H3 and it is verified. The prompt is given:
*"(1) Identify the essential problem in the post. (2) Think step by step to reason about what should
be included in the relevant documents. (3) Draft an answer."* Generators tested: GPT-4-0125-preview,
Llama-3-70B-Instruct, GritLM.

## 3. A tension our two source papers do not resolve

On BRIGHT, **BM25 benefits most** from reasoning queries — 14.5 → **27.0** with GPT-4 rewrites
(Table 38), the single best BM25 result in the paper and better than most dense models on original
queries. The authors' explanation: "BM25 can adapt to different queries, while LLM-generated queries
are out-of-distribution for trained models."

On TEMPO, the same manoeuvre **destroys** BM25: 10.8 → 4.3–6.2 (their Figure 8).

**Directly opposite conclusions about the sparse arm, from the two benchmarks that matter most to
us.** My earlier note stated "never rewrite the BM25 arm" as a rule; on this evidence that was
overconfident. It is a hypothesis to measure on RETECO data, not a design constraint. TEMPO is the
closer analogue (it *is* our data), so the prior still leans that way — but only leans.

## 4. THE CORRECTION — cross-encoder rerankers hurt here

Table 3, average reranking performance:

| Retriever | Reranker | k | nDCG@10 |
|---|---|---:|---:|
| BM25 | none | – | 14.3 |
| BM25 | MiniLM cross-encoder | 10 | 13.1 |
| BM25 | MiniLM cross-encoder | 100 | **8.3** |
| BM25 | Gemini-1.0 | 10 | 15.7 |
| BM25 | GPT-4 | 10 | 17.4 |
| Google | none | – | 19.5 |
| Google | MiniLM cross-encoder | 100 | **9.4** |
| Google | GPT-4 | 100 | 22.6 |

> "The traditional cross-encoder negatively impacts retrieval quality, with performance declining as
> more documents are reranked, suggesting that **training rerankers on MS MARCO does not transfer
> well to BRIGHT**."

`ms-marco-MiniLM-L-12-v2` at k=100 nearly **halves** BM25's score. LLM rerankers help, and help more
with stronger LLMs and larger k — the opposite direction.

**This contradicts what I wrote in `SUMMARY.md` last session**, which ranked "cross-encoder rerank of
top-100, +4–7" at #4 and named bge-reranker-v2-m3 — an MS MARCO-lineage model, at exactly the k that
does the most damage. `SUMMARY.md` is corrected accordingly. Phase 7 (H4) must test small k first
and treat any MS MARCO-trained reranker as presumed harmful until measured.

## 5. Why BRIGHT rank order does not transfer to TEMPO

DiVeR leads both, but below the top the orders invert: on TEMPO, E5 (30.4) and SFR (30.0) beat
ReasonIR (27.2); on BRIGHT, ReasonIR (24.4) comfortably beats E5 (17.9) and SFR (18.3). Both
benchmarks draw heavily on Stack Exchange, so this is not domain shift — plausibly the temporal
dimension rewards different behaviour. **Prefer TEMPO numbers where they exist** (TEMPO Table 3
covers 12 models); use BRIGHT only as a tiebreaker.

## 6. Two more results worth keeping

- **Robust to pretraining leakage.** Continuing to train GritLM on the benchmark's own Stack Exchange
  documents (LM loss + contrastive on QA pairs, without query→document mappings) moved the average
  from 20.5 to **20.4** — nothing. In-domain pretraining does not buy reasoning ability. Kills any
  idea of domain-adaptive pretraining on the RETECO corpus as a cheap win.
- **Long-context retrieval is harder still.** On unsplit web pages (avg up to ~40k tokens), best
  recall@1 is 27.8 (Qwen). Relevant to the Phase 2 document-length audit.
- RAG: stronger retrievers give better QA (Qwen +1.9 over closed-book) but oracle gives +4.1 —
  same "retrieval is the bottleneck, and QA does not cleanly measure it" caveat as TEMPO and RECOR.
