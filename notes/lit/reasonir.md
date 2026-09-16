# ReasonIR — highest ceiling under rewriting, but too big for a T4

**Shao, Qiao, Kishore, Muennighoff, Lin, Rus, Low, Min, Yih, Koh, Zettlemoyer** (FAIR at Meta / UW /
NUS / AI2 / Stanford / MIT / Berkeley). *ReasonIR: Training Retrievers for Reasoning Tasks.*
arXiv **2504.20595v1**, 29 Apr 2025. Read directly 16 Sept 2026.
Code <https://github.com/facebookresearch/ReasonIR> · Model <https://huggingface.co/reasonir/ReasonIR-8B>

First bi-encoder trained specifically for reasoning-intensive retrieval. **Llama-3.1-8B** fine-tuned with a
bi-directional attention mask and mean pooling over last-layer hidden states. Trained on public data
(1.38M) + synthetic varied-length (245k) + hard-query (100k) data with multi-turn-generated hard negatives.

## BRIGHT results

| Setting | nDCG@10 |
|---|---:|
| original query | 24.4 |
| + Llama-3.1-8B reason-query | 28.0 |
| + GPT-4 reason-query | 29.9 |
| + GPT-4 reason-query + BM25 hybrid (0.5) | 32.0 |
| + QwenRerank (Qwen2.5-32B, top-100) | **36.9** |

On TEMPO it scores 27.2 base, rising to **35.3** with "Normalized" temporal-tag queries (+8.0) and
**41.0** with GPT-4o reasoning augmentation (+13.7) — the largest rewriting gain of any model tested.

## Why we probably cannot use it as the first stage

8B in fp16 ≈ 16 GB. A T4 is 16 GB total, and a T4 is **compute capability 7.5 → no bf16**, so the usual
memory escape hatch is unavailable. Embedding 1.65M documents with it is not realistic in our budget.
DIVER-Retriever-4B beats it on BRIGHT original queries (28.9 vs 24.4) at half the size — strictly better
for us. Revisit only if quantised inference proves stable, which is a Phase 6 question at best.

## The finding that directly shapes H2

> "We evaluated a popular query decomposition method on BRIGHT and found that it **reduces**
> performance from 12.1 to 10.5 with Nomic and 20.4 to 17.3 with GRIT-7B... **an information-rich long
> query is better than several decomposed short queries.**"

Combined with TEMPO's Step-Only (14.6) ≪ Query+Step (26.4), two independent papers agree:
**decomposition used as a replacement for the query hurts; decomposition used to augment the full query
helps.** So H2 must be tested as *fusion of Query+Step rankings*, never as Step-Only retrieval. If we skip
this we will reproduce a known negative result and waste the phase.

## The other transferable idea — query-length scaling

ReasonIR keeps improving as rewritten queries grow from 64 → 2048 tokens, while GRIT-7B and Nomic
plateau or degrade (GRIT was trained with a 256-token query limit). Query length is a test-time scaling
axis, but **only for models trained for it**. Check the context limit of whatever embedder we pick before
generating long rewrites; a long rewrite into a short-context encoder is wasted compute.

Also useful: ReasonIR-8B + BM25 overlap in only 28.2% of top-100 docs — sparse and dense find
genuinely different documents, which is why hybrid fusion works.
