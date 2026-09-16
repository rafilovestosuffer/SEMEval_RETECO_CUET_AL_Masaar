# Starter-kit audit — what the organizers' code actually does

Read 16 Sept 2026 from `https://github.com/DataScienceUIBK/RETECO.git` at commit
**`23093c30e568a337c6293de4c58ffcccddbded5b`** (2026-08-30, "Add icons and an entrance to the News list").

Everything below is read off that code, not inferred from the task description. Anything I could not
confirm is marked **[UNVERIFIED]**.

> **Access note.** The RETECO GitHub repo *is* reachable from a Claude Code web session through the git
> proxy. `huggingface.co` is **not** — 403 on CONNECT, including `git ls-remote` against
> `tempo26/Tempo` and `DataScience-UIBK/RETECO-SemEval2027`. So the code can be read here; the data
> cannot.

---

## 1. The metric — corrects CLAUDE.md §4

**Macro-average is over the 13 domains, not over topics.** Two levels:

1. Within a domain, `pytrec_eval` scores each topic and `calculate_retrieval_metrics` divides by
   `len(scores)` — the topic mean.
2. Across domains, `official_baseline.py` computes `sum(v[m] for v in vals) / len(vals)` over domains,
   commented "macro-average over domains, as upstream run.py does".

`BASELINE_RESULTS.md` says the same in prose: "Macro-averaged over domains, as both papers do."

**Arithmetic check.** The 13 published per-domain 1a-train values
(0.0695, 0.1349, 0.0382, 0.1003, 0.0691, 0.1627, 0.0199, 0.0943, 0.0278, 0.2792, 0.0255, 0.0429, 0.0777)
sum to 1.1420, ÷13 = **0.08785** → rounds to the published **0.0879**. ✓

**Why it matters.** Every domain is worth exactly 1/13. History's 801 queries carry no more weight than
IOTA's ~10. Optimize and report per-domain; never pool topics globally. Small domains are also
high-variance (IOTA dev is ~3 queries), so Phase 3's folds must be stratified by domain with per-domain
CIs, and a single tiny domain swinging can move the macro number more than it should.

## 2. There are two BM25 baselines and they are not interchangeable

| | Tokenisation | Scoring | Reproduces `BASELINE_RESULTS.md`? |
|---|---|---|---|
| `official_baseline.py` | pyserini Lucene analyzer (Porter stemming + English stopwords), gensim `LuceneBM25Model` k1=0.9 b=0.4, top-1000 | `pytrec_eval` `ndcg_cut_10` | **Yes** |
| `bm25_baseline.py` + `bm25.py` | regex `[A-Za-z0-9]+`, lowercased, no stemming, no stopwords, BM25+ style non-negative idf | pure-Python `ir_metrics.py` | **No** — its own README calls it "approximate" |

The starter-kit README is explicit: "**nDCG@10 from `pytrec_eval` is the official RETECO metric**."

So the Phase 1 gate requires the pyserini path, which requires a **JDK** (21 works). That is the whole
reason Phase 1 runs on Kaggle Linux rather than the Windows laptop.

`scorer.py` (the pure-Python one) is *not* the official scorer. It is the zero-install approximation,
and it is also where the temporal diagnostics (TP@k, TR@k, TC@k, NDCG|FC@k) live — those are RETECO
additions, not part of the upstream retrieval metrics, and not the leaderboard metric.

## 3. Exact query construction — copy this, do not improvise

| Sub-track | Query string | Topic id |
|---|---|---|
| 1a | `examples_{split}.jsonl` → `record["query"]` | `record["id"]` |
| 1b | `f"{record['query']}\n\nStep: {step['step_instruction']}"` | `step["step_id"]` |

**Note `step_instruction`, not `step`.** `official_baseline.py`'s own docstring says "step_text" but the
code uses `step_instruction`; the code wins.

Corpus id field is `id` for Track 1 (`doc_id` for Track 2). Documents are `documents.jsonl` with keys
`content` and `id`. Qrels are `qid 0 docid rel`, space-separated, and **all positives are grade 1** —
binary. So graded-vs-binary nDCG is a non-issue here; `pytrec_eval`'s gain formula and a binary one agree.

## 4. Subtle behaviours that will silently change a number

- **`restrict_to_corpus`** — gold ids naming a document absent from the corpus are dropped, and a topic
  left with no surviving gold is dropped *entirely*. Then `scores` is filtered to topics present in
  `gt`. Net effect: reported `num_topics` can be lower than the query count, and the denominator of the
  topic mean changes with it. Do not assume `num_topics == len(examples)`.
- **Tie-breaking** — `sorted(zip(doc_ids, sims), key=lambda x: x[1], reverse=True)` is a stable sort on
  *corpus insertion order*, not on doc id. Two documents with identical scores rank by their position in
  `documents.jsonl`. Any reimplementation that breaks ties by doc id will diverge on tied scores.
- **Index reuse** — the corpus is analysed and indexed once per domain and shared across train and dev.
  The organizers flag this as their only deviation from upstream, affecting wall clock only.
- **Run files are a by-product** — scoring happens on the in-memory top-1000 dict; the written `.trec`
  file is truncated to top-100 and is tab-separated. Do not score the file and expect the same number.

## 5. Licensing — do not vendor this code

The RETECO repo has **no LICENSE file**. Its README says "A repository-level software license will be
added before the first tagged RETECO code release." So the starter-kit code is effectively
all-rights-reserved and must not be copied into this repository.

Our approach: clone it at the pinned commit `23093c3` at runtime and invoke it. No redistribution,
and the pin satisfies CLAUDE.md §5.7.

Data licences are separate and already in CLAUDE.md §4: corpus/Q&A text CC BY-SA 4.0 (Stack Exchange
origin, share-alike, cannot be relicensed); RETECO annotations (splits, qrels, `split_manifest.json`)
CC BY 4.0.

## 6. Provenance pins, for §5.7 and the paper

From `docs/sample_data/manifest.json` (pilot package dated 2026-07-29):

| Source | HF repo | Pinned revision | Paper |
|---|---|---|---|
| TEMPO | `tempo26/Tempo` | `f9df06c05688225e37701974d23c8e3c5d4efaf6` | arXiv 2601.09523 |
| RECOR | `RECOR-Benchmark/RECOR` | `d9faa639019dcfa1a1fea2aece55ebcba3083c00` | arXiv 2601.05461 |

Note the HF org for TEMPO is `tempo26`, while the GitHub repo in CLAUDE.md §4 is `tempo-bench/Tempo`.

**Both arXiv ids VERIFIED 16 Sept 2026** (read directly via alphaXiv; the earlier [UNVERIFIED] mark is
cleared):
- 2601.09523v1 — *TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive
  Retrieval*, Abdallah, Ali, Abdul-Mageed, Jatowt (Innsbruck + UBC), 14 Jan 2026.
- 2601.05461v1 — *RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark*, Ali,
  Abdallah, Agarwal, Patel, Jatowt (Innsbruck + Oracle AI), 9 Jan 2026.

Notes in `notes/lit/tempo.md` and `notes/lit/recor.md`. RECOR independently confirms the per-domain
macro averaging: "an average within each domain, then macro-average across all 11 domains to ensure
equal weight regardless of domain size."

`download_raw.py` pulls those two raw releases; `build_release.py` then constructs `reteco_data/`.
For Phase 1 we want the *already built* release from `DataScience-UIBK/RETECO-SemEval2027`, not a rebuild.

## 7. The sample data is not a usable test fixture

`docs/sample_data/track1_tempo/` looks inviting — 5 examples, 26 documents, qrels — but it is not
usable for wiring tests:

- Its `steps.jsonl` is **flattened**: top-level keys are `gold_ids`, `id`, `query`, with **no `steps[]`
  array**, so `topics_1b` (which iterates `record["steps"]` for `step_id`/`step_instruction`) cannot
  read it. Different schema from the real `steps_{train,dev}.jsonl`.
- Its `documents.jsonl` contains **only the 26 gold passages**. Its own README says: "Retrieval
  baselines and official scoring must use the full track corpus, not rank only these positive
  documents." Ranking a pool of pure positives yields meaningless nDCG.

So we use a synthetic fixture in the real release schema instead, and the sample data only as a
reference for field names and the qrels format.

## 8. Toolchain, exercised locally 16 Sept 2026

The full official stack was run end to end in a Claude Code container (JDK 21) against a
**synthetic** domain in the release schema (`tests/fixtures/make_fixture.py`, 300 docs, 12 queries):

- `pyserini` Lucene analyzer works under **JDK 21**. Confirmed it stems and drops stopwords:
  `"In 2023 the consensus protocols were revised and upgraded"` →
  `['2023', 'consensu', 'protocol', 'were', 'revis', 'upgrad']`. Note `were`/`what`/`about`
  survive — Lucene's default English stopword set is small.
- `official_baseline.py --track1 iota --track2 --splits train dev` exits 0 and writes
  `results.json` + four `.trec` files. Passing `--track2` with no values does skip Track 2.
- The emitted run files pass the organizers' own `format_checker.py`: *800 lines, 8 topics,
  0 errors, 0 warnings, VALID*.
- `eval/gate.py` parses that `results.json` correctly and returns the expected FAIL.

**The numbers from this run are meaningless** — the data is random. What is established is that
the stack executes and that our wrapper reads its output. Only the real IOTA domain closes the gate.

Two install findings, both folded into the Phase 1 kernel:

- **Use `pytrec-eval-terrier`, not `pytrec_eval`.** The starter kit's `requirements.txt` names
  `pytrec_eval>=0.5`, which compiles trec_eval from source and **downloads it during the build** —
  that fails behind any egress restriction (403 here). `pytrec-eval-terrier==0.5.10` ships wheels
  and imports as `pytrec_eval` with the same API.
- `pyserini` is heavy (pulls torch/faiss) and timed out on the default pip timeout; the kernel uses
  `--timeout 120 --retries 5`. For analyzer-only use, `pip install --no-deps pyserini pyjnius` is
  enough and pulls just the `anserini-1.7.1-fatjar.jar`. *(A `Cannot uninstall PyJWT` error seen
  here is a Debian-packaged-PyJWT artifact of this container, not a pyserini problem, and will not
  occur on Kaggle.)*

## 9. Open questions this did *not* answer

- The internal file layout of `DataScience-UIBK/RETECO-SemEval2027` — HF is unreachable here. The Phase 1
  kernel probes it with `list_repo_files` and prints the tree before downloading anything.
- Whether the Kaggle image ships a JDK new enough for pyserini. The kernel probes and reports.
- Whether the hidden test set draws from the same 13 domains — still an organizer question (CLAUDE.md §4).
- Team/submission caps — still unannounced.
