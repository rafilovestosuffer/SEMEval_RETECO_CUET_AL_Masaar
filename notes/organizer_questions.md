# Questions for the RETECO organizers

Draft for the §12 TODO. Send to the mailing list (`semeval-2027-reteco`) rather than direct
mail where possible, so the answers reach every team — several of these affect everyone.

Contact: abdelrahman.abdallah@uibk.ac.at · Team: `CUET_AL_Masaar`

**Before sending, re-check the site** — <https://datascienceuibk.github.io/RETECO/participate.html>
and the evaluation page. Some of these may have been answered since 16 Sept 2026, and asking
something already documented wastes the organizers' time and our credibility.

---

## Resolved — do NOT ask

- ~~Do the hidden test queries come from the same 13 domains?~~ **Answered on the site**: the test
  set is "stratified across all 13 TEMPO domains", ~350 Track 1 test queries, and "the corpus is
  never split". So per-domain tuning is safe on coverage grounds; the residual risk is variance at
  roughly 27 test queries per domain.

## ASK FIRST — the two official documents disagree

- **Averaging level of the official metric.** This was marked resolved on 16 Sept and reopened on
  22 Sept; it is now the highest-value question we have, because the two answers differ by 6× in
  how much the History domain is worth.
  - `evaluation.html` says the leaderboard is a macro over **queries**: 1a "nDCG@10 is computed
    independently for each query and macro-averaged", 1b averages a query's steps then aggregates
    across queries, and per-domain results are "reported **diagnostically**".
  - `BASELINE_RESULTS.md` and `official_baseline.py:259` compute a macro over **domains**
    (`sum(vals)/len(vals)`, `num_topics` excluded) — which is why the 13 published per-domain
    1a-train values average to 0.08785 → the published 0.0879.
  - Suggested wording: *"The evaluation page describes nDCG@10 for 1a as macro-averaged over
    queries, while `official_baseline.py` aggregates the baseline table as an equal-weight mean
    over the 13 domains. Which aggregation determines the official ranking? And for 1b, is a
    query with eight steps weighted the same as a query with two?"*
  - Why it matters, stated plainly: under the query macro, History is ~46% of train+dev; under the
    domain macro it is 7.7%. Systems tuned for one can lose under the other.

- **Does doc2query-style index enrichment count as "augmenting the corpus"?** The rules forbid
  external corpora replacing or augmenting the official one, but generating expansions *from* the
  official documents is not obviously external. Worth confirming before Phase 4.

- **Is the evaluation-phase index byte-identical to the released corpus?** Decides whether
  embeddings computed now can be reused in January, which is a multi-week compute commitment.

---

## Draft

> **Subject:** SemEval-2027 Task 1 (RETECO) — questions on test data, submission limits and reporting
>
> Dear organizers,
>
> I am preparing a Track 1 (sub-tracks 1a and 1b) system for SemEval-2027 Task 1 and have a few
> questions that affect how I design and validate it. Apologies if any are already answered
> somewhere I have missed.
>
> **1. Test data provenance.** Will the hidden test queries be drawn from the same 13 TEMPO
> domains and the same corpora as the train/dev release, or should we expect unseen domains?
> This determines whether per-domain configuration is meaningful or whether everything must be
> domain-agnostic — a fairly fundamental design decision given the metric is macro-averaged
> over domains.
>
> **2. Test corpus.** Will retrieval at test time run over the same per-domain corpora already
> released, or will a new or extended corpus be provided? If the corpora are unchanged, we
> would like to confirm that document embeddings computed now remain valid for the evaluation
> window.
>
> **3. Submission limits.** Once the evaluation platform opens, what are the limits on
> submissions per team per day, and on total submissions? Is more than one run per sub-track
> permitted, and if so which is scored?
>
> **4. Team registration.** Are there constraints on team size, or on one person participating
> in more than one SemEval-2027 task? I am a solo participant and also intend to enter Task 6.
>
> **5. Hardware and compute reporting.** The system-paper checklist mentions reporting compute.
> Is there a required format or a minimum level of detail (GPU type and hours, wall clock,
> model sizes)? I am working on free-tier Kaggle GPUs and would like to report this accurately.
>
> **6. Paper timeline.** The site marks the February 2027 system-paper deadline, March
> notification and April camera-ready as tentative. Are these now confirmed, and will there be
> a separate call with formatting requirements?
>
> **7. Temporal diagnostics at test time.** `scorer.py` computes Temporal Precision, Temporal
> Relevance and Temporal Coverage using the step files as the source of required periods, and
> the docstring notes that on the hidden test these come from an LLM-as-judge. Will those
> diagnostic figures be reported back to participants alongside nDCG@10? They would be valuable
> for the system paper's analysis even though they are not the ranking metric.
>
> Thank you for assembling this task — the starter kit and the published per-domain baselines
> have made getting started unusually straightforward.
>
> Best regards,
> Rafiur Rahman (CUET_AL_Masaar)
> Chittagong University of Engineering & Technology, Bangladesh

---

## Why each question is worth asking

| # | Decision it unblocks |
|---|---|
| 1 | Whether per-domain tuning is legitimate at all. Under an equal-weight domain macro, unseen test domains would make per-domain choices actively harmful. |
| 2 | Whether Phase 4's embedding cache survives to January. Re-embedding 1.65M documents is ~6–10 T4-hours we cannot spend twice. |
| 3 | Phase 9 planning; §5.1 forbids anything that looks like working around limits, so we need the real numbers. |
| 4 | §5.1 — one account, one team. Confirming the rules in writing protects against an accidental violation. |
| 5 | §10 Phase 10 requires compute reporting in the paper; better to collect it in the right format from Phase 4 onward than reconstruct it in February. |
| 6 | Scheduling against the Task 6 and BD-BrowseComp deadlines (§2). |
| 7 | Whether the temporal diagnostics can be part of the paper's analysis, or whether we must compute proxies ourselves on dev. |

## Still open after this

Nothing that the organizers can answer. The remaining unknowns — document length distribution,
what `guidance_*.jsonl` actually contains, whether it carries TEMPO's reasoning-class labels
(H6) — are answered by Phase 2's audit of data we already have, not by asking.
