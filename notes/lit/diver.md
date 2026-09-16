# DIVER — best system on TEMPO, and the one that fits our GPU

**Sun, Long, Yang, Wang et al. (Ant Group + Sun Yat-sen).** *DIVER: A Multi-Stage Approach for
Reasoning-intensive Information Retrieval.* arXiv **2508.07995v5**, latest revision 2 Apr 2026.
Code <https://github.com/AQ-MedAI/Diver>. Read directly 16 Sept 2026.

Top scorer on TEMPO (32.0 nDCG@10, Table 3 there) and SOTA on BRIGHT (46.8). Four stages:

1. **DIVER-DChunk** — rule-based cleaning (blank lines, truncated sentences) + semantic rechunking to
   ≤4k tokens with 20% overlap; chunks keep the *original* doc id and a doc scores as the **max** over its
   chunks. Worth +0.5 avg for the dense retriever, ~0 for BM25.
2. **DIVER-QExpand** — iterative LLM query expansion with corpus feedback (based on ThinkQE). Two
   rounds, top-5 docs each, docs truncated to 512 tokens, Qwen-R1-Distill-14B at temp 0.7. Keeps only the
   original query + final-round expansion (not all intermediates) to control length.
3. **DIVER-Retriever** — **Qwen3-Embedding-4B** fine-tuned with InfoNCE on ~220k synthetic
   medical/general/math examples with hard negatives. EOS-token last-layer embedding, cosine similarity.
4. **DIVER-Rerank** — pointwise LLM helpfulness score 0–10 (Qwen2.5-32B-Instruct), interpolated
   `0.6·reranker + 0.4·retriever`, then combined with a listwise reranker over top-100.

## The decisive practical fact

> **Weights are open: `AQ-MedAI/Diver-Retriever-4B-1020` on Hugging Face.**

**4B in fp16 ≈ 8 GB → fits a T4 (16 GB) with headroom.** ReasonIR-8B (Llama-3.1-8B) is ~16 GB fp16 and
does **not** comfortably fit. On BRIGHT original queries DIVER-Retriever (28.9 v1 / 31.9 v2) already beats
ReasonIR-8B (24.4) and RaDeR-7B (25.5) while being half the size. For our compute budget this is the
clear first-stage candidate.

The reranker (`Diver-GroupRank-32B`) is also open but 32B — **out of reach on a T4**; use a small
cross-encoder instead at Phase 7.

## Hybrid recipe — take this verbatim

`S = 0.5·S_dense + 0.5·S_BM25`, **both min-max normalised to [0,1] first**. On BRIGHT this moves
DIVER-Retriever 33.9 → **37.2** (+3.3) and ReasonIR 32.6 → 35.7 (+3.1). Cheap, consistent, no training.
This is the Phase 4 hybrid baseline; RRF is the alternative to compare against.

## Warning that will save us a GPU week

**Off-the-shelf Qwen3-Embedding-4B scores 5.6 on BRIGHT** (Table 3) — near-random, worse than BM25's
14.5. The same backbone fine-tuned becomes 28.9. A strong MTEB/BEIR score predicts nothing here:
SFR-Embedding-Mistral is 59.0 on BEIR and 18.3 on BRIGHT. **Never pick an embedder off MTEB for this
task.** Use models with published BRIGHT *or* TEMPO numbers only.

## Cost to be honest about

Stages 2 and 4 both need an LLM at query time; stage 2 additionally needs two retrieval rounds per query.
That is a real per-query cost on a hidden test set of unknown size, and it is why §9's Phase 6 is
cost-gated. Stages 1 and 3 are one-off / offline and are the parts we should take first.
