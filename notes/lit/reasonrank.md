# ReasonRank — the Phase 7 reranker

**Liu, Ma, Sun, Zhu, Li, Yin, Dou (Renmin Univ. + Baidu + CMU).** *ReasonRank: Empowering
Passage Ranking with Strong Reasoning Ability.* arXiv **2508.07050v3** (22 Apr 2026). Code
<https://github.com/8421BCD/ReasonRank> (read at `15d8733`). Weights
`liuwenhan/reasonrank-7B` (MIT, Qwen2.5-7B-Instruct base, revision `3444046`). Read directly
23 Sept 2026.

Listwise reasoning reranker: the model reads a window of 20 passages, writes a `<think>` chain
(~630–930 tokens on BRIGHT, Fig. 13), then an `<answer>` permutation. Trained on 13K
DeepSeek-R1-labelled examples (SFT, then GRPO with an NDCG@10 + Recall@10 + RBO reward).

## Why it and not a cross-encoder — BRIGHT Table 1, reranking ReasonIR-8B's top-100

| reranker | BRIGHT avg nDCG@10 | vs retriever 30.59 |
|---|---:|---:|
| RankT5 (3B) | 16.60 | −14.0 |
| Rank-R1 (7B) | 20.57 | −10.0 |
| RankZephyr (7B) | 22.64 | −8.0 |
| Rank1 (7B) | 27.23 | −3.4 |
| Rearank (7B) | 31.75 | +1.2 |
| **ReasonRank (7B)** | **35.74** | **+5.2** |
| ReasonRank (32B) | 38.03 | +7.4 |

The authors' own reading: "except for Rank-K (32B), [baselines] can hardly improve the initial
retrieval results." That is the same failure BRIGHT reported for MS MARCO cross-encoders
(`bright.md`), and it is why a small generic reranker is not the default here. Their ablation
also shows training on MS MARCO alone costs **5.66** points (35.74 → 30.08).

Reranking only the **top-20** (Table 9) still gives 30.59 → **33.23** (+2.6) at a fraction of
the cost, which is the evidence for a shallow depth under our quota.

## Settings we copy verbatim (their `run_rank_llm.sh`, BRIGHT)

window 20, step 10, passages truncated to **512 tokenizer tokens**, `temperature=0`,
`max_tokens = 3072 + 100`, system prompt and prompt text from `listwise_prompt_r1.toml` /
`utils.py`, `[n]` → `(n)` in query and passages. Ported in `reteco/rerank.py`.

## Cost and fit

7.6B params: ~15.2 GB in fp16 — does **not** fit one 16 GB T4 with a KV cache, so it runs with
vLLM tensor parallelism over Kaggle's T4 x2. The checkpoint is bf16; the T4 has no bf16, so it
is loaded as fp16 (same risk as DIVER, `diver.md`). The paper reports the listwise model is
2–2.7x faster than pointwise Rank1-7B (Fig. 5) because it writes one reasoning chain per
window, not per passage.

## Caveats

- BRIGHT numbers use GPT-4-rewritten queries for *retrieval* and the original query for
  reranking; our first stage uses the original queries throughout.
- No TEMPO numbers are reported. Whether its gains transfer to temporal questions is exactly
  what Phase 7 measures.
- The 32B model is out of reach on our hardware.
