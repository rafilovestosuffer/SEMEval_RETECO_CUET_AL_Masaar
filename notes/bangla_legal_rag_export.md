# Export for the Bangla Legal RAG project

CLAUDE.md §1 asks that every RETECO component be reusable for multi-hop retrieval over amended
Bangla statutes. This note lists what transfers unchanged, what transfers with a swap, and which
findings are worth re-testing there. Every number cited is from `results/ledger.csv` and applies
to RETECO Track 1 (English, StackExchange). None of it has been measured on Bangla legal text.

## Modules that transfer as they are (language-agnostic, unit-tested)

| module | what it gives you | tests |
|---|---|---|
| `reteco/stepfuse.py` | fuse a question's ranking with its sub-question rankings (sum / max / RRF; theoretical, min-max or rank normalisation; corpus-order tie-break) | `tests/test_stepfuse.py` |
| `reteco/fusion.py` | generic RRF and weighted/max/sum score fusion | `tests/test_fusion.py` |
| `reteco/dedup.py` | content-hash deduplication before embedding, then scatter back to every id | `tests/test_dedup.py` |
| `reteco/runs.py` | TREC run read/write with a deterministic, stable tie-break | `tests/test_data_runs.py` |
| `reteco/rerank.py` | listwise sliding-window reranking: prompt, window plan, permutation parser with repair | `tests/test_rerank.py` |
| `eval/score.py`, `eval/bootstrap.py`, `eval/cv.py` | per-query nDCG via pytrec_eval, stratified folds, paired bootstrap CIs | `tests/test_score.py`, `tests/test_cv_bootstrap.py` |
| `submit/assemble_runs.py` | "never drop a topic" merge of a partial expensive run with a cheap complete one | `tests/test_assemble_runs.py` |

## What needs a swap

- **Encoder.** Diver-Retriever-0.6B is trained on English reasoning data. For Bangla, pick an
  encoder with measured Bangla retrieval numbers; the RETECO lesson is not to pick on MTEB
  English averages (`notes/lit/SUMMARY.md`, fact 2).
- **Reranker prompt.** `reteco/rerank.py` holds ReasonRank's English prompt verbatim, because a
  listwise model is trained on one prompt. A Bangla reranker needs its own; the window plan and
  parser do not change.
- **Decomposition.** RETECO supplies gold-free official steps. A legal system has to generate its
  sub-questions (e.g. one per amendment or effective date), and their quality is a new variable.

## Findings worth re-testing there

1. **Merge candidate pools; don't blend scores** (Phase 5, 5b). Fusing a question with its
   decomposed steps gained +0.0115 nDCG@10, all of it from putting the whole question's
   candidates into the pool. A parent weight of 0.001 captured the full gain. Operator choice
   made no significant difference. Amended-statute chaining is the same shape (one question, several
   version-specific sub-questions), so start from the union and a tiny parent weight.
2. **The union helps one direction only** (Phase 5c). Adding the question's candidates to each
   sub-question's list did not help sub-question retrieval, and hurt at larger weights. Keep
   per-sub-question retrieval separate.
3. **Weak lexical retrieval adds little once dense retrieval is strong** (Phase 4). BM25 found
   only 104 relevant documents that dense missed, against 1,706 the other way. Measure this
   one-directional overlap before building a hybrid, because legal text may differ (exact section
   numbers favour lexical matching).
4. **A reasoning reranker was the second-largest gain** (+0.0508, Phase 7), but it cost 30 s per
   query on two T4s. Budget for reranking by query count, and use `assemble_runs.py`-style
   fallback so a cut-off session still yields a complete run.
5. **Deduplicate before embedding.** 29.4% of RETECO's corpus was byte-identical. Statute
   collections with repeated consolidated versions are likely to be similar; this is the cheapest
   GPU saving available.
