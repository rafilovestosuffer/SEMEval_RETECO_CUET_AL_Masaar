# Track 1 data audit — Phase 2 deliverable

Produced by `kaggle/kernels/phase2_bm25_full` on 2026-09-22 (kernel
`reteco-phase2-bm25-full-track1`), raw output in `cache/p2/data_audit.json`. The guidance
analysis below was extended locally afterwards — see *Blind spot* at the end.

Corpus: **1,654,055 documents / 4.44 GB** across 13 domains, matching the published figure.
Splits match the official table exactly: 1a 1,211 train / 519 dev queries, 1b 2,762 / 1,214 steps.

---

## 1. The finding that changes indexing: 29% of the corpus is duplicate text

| domain | docs | duplicate-text docs | share |
|---|---:|---:|---:|
| history | 356,493 | 155,842 | **43.7%** |
| bitcoin | 153,291 | 76,515 | 49.9% |
| politics | 183,394 | 47,993 | 26.2% |
| hsm | 213,818 | 44,199 | 20.7% |
| cardano | 87,201 | 39,678 | 45.5% |
| monero | 85,093 | 31,070 | 36.5% |
| workplace | 64,659 | 21,000 | 32.5% |
| genealogy | 156,228 | 20,037 | 12.8% |
| economics | 93,756 | 18,989 | 20.3% |
| travel | 177,677 | 18,846 | 10.6% |
| law | 43,288 | 7,116 | 16.4% |
| quant | 28,785 | 3,706 | 12.9% |
| iota | 10,372 | 692 | 6.7% |
| **total** | **1,654,055** | **485,683** | **29.4%** |

Measured as MD5 of the exact `content` string, so this is byte-identical duplication, not near
duplication — the real figure under fuzzy matching can only be higher. **Zero duplicate ids and
zero empty documents**, so the duplication is many ids pointing at identical text.

Why it matters, in order of how much it costs:

- **Phase 4 embedding cost.** Embedding is the dominant GPU expense and 29% of it is redundant.
  Embed unique content hashes once and map hashes back to doc ids: a ~29% saving on the single
  most expensive operation in the project, and more on History, which is both the largest domain
  and the most duplicated.
- **Ranking.** Duplicates compete for the same top-10 slots. If several ids share text, they get
  near-identical scores and can occupy multiple ranks, pushing distinct evidence out of the cut.
  nDCG@10 only credits documents that are *in the qrels*, so a duplicate of a gold document that
  is not itself gold is a pure loss. This is worth an explicit dedup-before-cut experiment.
- **It does not affect the BM25 gate.** The reproduction matched with duplicates present, because
  the organizers' baseline indexes them too. Any dedup is our deviation and must be ablated.

## 2. Document length: a sane median with a pathological tail

Median document is **104–163 whitespace tokens** depending on domain — short. But the tail is
extreme: p99 ranges from 1,271 (travel) to 28,271 (iota) words, and the single longest document
in bitcoin is **1,513,968 words**. History's p99 is 7,264 against a median of 104.

Consequences:
- Truncation is nearly free at the median and catastrophic at the tail. A 512-token cap touches
  under 10% of documents in most domains.
- DIVER's DChunk (rechunk to ≤4k tokens, score a doc as the max over its chunks) is aimed exactly
  at this shape. With p90 under ~615 words everywhere, chunking only matters for a thin slice —
  but that slice includes documents that a naive truncation would destroy.
- Watch for memory blowups: a single 1.5M-word document will not fit a tokenizer call naively.

## 3. Gold and steps

- **Gold per query (1a):** History median 3, mean 4.5, p90 9, max 39. Multi-gold is the norm, so
  recall@k matters, not just rank-1.
- **Steps per query (1b):** History median 3, p99 4, max 5. Step fusion therefore combines a
  handful of lists per query, not dozens — cheap, and consistent with the H2 plan.
- **Doc ids** are `<domain>/<hash>_<number>.txt`, e.g. `history/930b6ad6_136585.txt`. Not numeric,
  so never sort or compare them as integers.

## 4. Zero unreachable gold — `restrict_to_corpus` is a no-op here

| | topics | unreachable gold ids | topics fully dropped |
|---|---:|---:|---:|
| 1a train | 1,211 | **0** | 0 |
| 1a dev | 519 | **0** | 0 |
| 1b train | 2,762 | **0** | 0 |
| 1b dev | 1,214 | **0** | 0 |

Every gold id names a document that exists in its domain corpus. **This corrects a claim made
earlier on 2026-09-22**, when Phase 1's `num_topics=7` for IOTA 1a train was read as
`restrict_to_corpus` dropping 3 of 10 queries. It does not: IOTA simply *has* 7 train and 3 dev
queries. `reteco.data.restrict_to_corpus` is therefore **still unexercised on real data** — it is
correct as written against the organizers' code, but Phase 2 gave it nothing to do and must not be
cited as having validated it.

## 5. `guidance_*.jsonl` — H6 is answerable, and it is richer than expected

Present for all 26 domain/split combinations, 1,730 records, three fields: `id`,
`query_guidance`, `gold_passage_annotations`.

`query_guidance` is a dict with twelve keys:

```
expected_granularity      is_temporal_query        key_time_anchors
quality_checks            query_summary            query_temporal_events
query_temporal_signals    retrieval_plan           retrieval_reasoning
temporal_intent           temporal_reasoning_class_primary
                          temporal_reasoning_class_secondary
```

**`temporal_reasoning_class_primary` answers H6's precondition.** Distribution over the 1,211
*train* queries (dev deliberately not reported — §5.2):

| class | n | share |
|---|---:|---:|
| event_analysis_and_localization | 435 | 35.9% |
| time_period_contextualization | 256 | 21.1% |
| origins_evolution_comparative_analysis | 171 | 14.1% |
| **trends_changes_and_cross_period** | **117** | **9.7%** |
| event_verification_and_authenticity | 80 | 6.6% |
| materials_artifacts_and_provenance | 68 | 5.6% |
| *(empty string)* | 36 | 3.0% |
| causation_analysis | 15 | 1.2% |
| sources_methods_and_documentation | 15 | 1.2% |
| historical_attribution_and_context | 15 | 1.2% |
| `none` / 2 singletons | 3 | 0.2% |

Three things follow, and they cut against the obvious plan:

1. **TCP is only 9.7% of train.** TEMPO reports Trends & Cross-Period as its hardest class
   (~17.9 mean nDCG@10), and it is the reasoning RETECO is named for — but it is a small minority
   of the actual queries. A system specialised for TCP optimises under a tenth of the score.
2. **TCP is wildly unevenly distributed**: economics 55.2%, quant 29.2%, politics 28.6%, iota
   28.6% — against history 2.9%, travel 1.4%, genealogy 2.5%. Since History is ~46% of the
   query-macro (§7 below), **the dominant domain is dominated by the easier classes.**
3. **61 of 1,211 train queries (5.0%) have `is_temporal_query: false`.** A temporal benchmark
   containing non-temporal queries is worth a sentence in the paper, and worth checking whether a
   time-aware method hurts them.

Also present and not yet exploited:
- `retrieval_plan` is a list of `{step, action, gold_ids}` — a **per-step gold mapping**, i.e. a
  supervision signal for exactly the step→document association H2 is about. Train split only
  (§5.4), and never at inference.
- `gold_passage_annotations` carries per-document `tense_guess`, `time_mentions`,
  `time_scope_guess` (`start_iso`/`end_iso`/`granularity`) and a `confidence`. Many ISO fields are
  empty strings in the sample inspected, so measure fill rate before depending on them.

## 6. Blind spot in the automated audit — fixed for next time

The kernel scanned guidance for TEMPO's class *abbreviations* (`TCP`, `HAC`, `CAU`) with a word
boundary regex and reported **NONE**, which read as "H6 is unanswerable". That was wrong: the
release spells the classes out (`trends_changes_and_cross_period`), and the audit's
`small_value_counts` only expanded *top-level* string fields, so a class buried one level inside
the `query_guidance` dict was invisible to both checks. The distribution above came from a local
pass over the guidance files afterwards.

Lesson worth keeping: an automated audit that reports an absence is only as good as the shape it
expected. The nested-dict expansion is the fix.

## 7. Cross-reference: which metric this is all weighted by

Under the leaderboard's query-macro (CLAUDE.md §4), History is **46.3%** of 1a train (561/1211)
against 7.7% under the domain-macro. Measured BM25 gap between the two aggregations:

| | domain-macro | query-macro | delta |
|---|---:|---:|---:|
| 1a train | 0.0879 | 0.0944 | +0.0066 |
| 1a dev | 0.0967 | 0.1055 | +0.0088 |
| 1b train | 0.0852 | 0.0910 | +0.0059 |
| 1b dev | 0.1063 | 0.1151 | +0.0088 |

7–9% relative — large enough that a reported gain can change sign between the two. Report both.
