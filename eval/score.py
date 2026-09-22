#!/usr/bin/env python3
"""Scoring: `pytrec_eval` nDCG@10, under **both** aggregations RETECO uses.

**Corrected 22 Sept 2026.** This module previously asserted that the equal-weight mean over the
13 domains was "the official metric" and that pooling over topics was "not". That was backwards
for the purpose that matters. There are two aggregations and they answer different questions
(CLAUDE.md §4):

1. **`pooled_over_topics` — the LEADERBOARD metric.** `evaluation.html`: 1a nDCG@10 is "computed
   independently for each query and macro-averaged"; 1b averages over a query's steps first, then
   across queries. Per-domain figures are "reported diagnostically". **This is what ranks us, so
   this is what to optimize.**
2. **`macro_over_domains` — the organizers' BASELINE TABLE metric.** `BASELINE_RESULTS.md` says
   "Macro-averaged over domains", and `official_baseline.py:259` computes `sum(vals)/len(vals)`
   over per-domain entries with `num_topics` excluded. Reproducing it is exactly what the Phase 1
   and Phase 2 gates check, so it stays — but it is a wiring check, not the objective.

They differ a lot, because domain sizes differ by two orders of magnitude: History has 801
queries and IOTA about 10, so History is ~46% of the pooled score and 1/13 of the domain macro.
A change that moves one can move the other the opposite way.

**Report both on every run.** It costs nothing and it is the only way to notice when a gain is an
artefact of the aggregation rather than a real improvement.

Usage::

    python eval/score.py --run run.txt --qrels qrels.txt --domain iota
    python eval/score.py --runs-dir results/runs --data reteco_data/track1_tempo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reteco.data import load_qrels, restrict_to_corpus  # noqa: E402
from reteco.runs import read_run_with_scores  # noqa: E402

__all__ = [
    "K_VALUES",
    "OFFICIAL_METRIC",
    "per_topic_scores",
    "domain_score",
    "macro_over_domains",
    "pooled_over_topics",
]

# The organizers request these cuts; nDCG@10 is the leaderboard metric.
K_VALUES = (1, 5, 10, 25, 50, 100)
OFFICIAL_METRIC = "ndcg_cut_10"


def _evaluator(qrels: dict[str, dict[str, int]]):
    """Build a pytrec_eval evaluator for the metrics the organizers report.

    Imported lazily so the rest of this module (and the tests for the averaging logic) work
    without pytrec_eval installed. Install `pytrec-eval-terrier`, not `pytrec_eval` — the
    latter builds trec_eval from source and downloads it at build time
    (`notes/starter_kit_findings.md` §8).
    """
    import pytrec_eval

    ks = ",".join(str(k) for k in K_VALUES)
    return pytrec_eval.RelevanceEvaluator(
        qrels, {f"ndcg_cut.{ks}", f"recall.{ks}", f"map_cut.{ks}", f"P.{ks}", "recip_rank"}
    )


def per_topic_scores(run: dict[str, dict[str, float]],
                     qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Score each topic. Returns ``{topic_id: {metric: value}}``.

    Only topics present in *both* the run and the qrels are scored, matching
    `official_baseline.py`, which filters the run to topics with surviving gold.
    """
    scoreable = {t: docs for t, docs in run.items() if t in qrels}
    if not scoreable:
        return {}
    return _evaluator(qrels).evaluate(scoreable)


def domain_score(run: dict[str, dict[str, float]],
                 qrels: dict[str, dict[str, int]],
                 corpus_ids: set[str] | None = None,
                 metric: str = OFFICIAL_METRIC) -> dict[str, float]:
    """Level 1: mean over topics within one domain.

    Args:
        corpus_ids: when given, `restrict_to_corpus` is applied first — unreachable gold is
            dropped and topics left with no gold disappear, which changes the denominator.

    Returns:
        ``{"score": mean, "num_topics": n}`` plus the mean of every other metric computed.
    """
    if corpus_ids is not None:
        qrels = restrict_to_corpus(qrels, corpus_ids)

    scores = per_topic_scores(run, qrels)
    if not scores:
        return {"score": 0.0, "num_topics": 0}

    metrics = sorted({m for s in scores.values() for m in s})
    means = {m: sum(s.get(m, 0.0) for s in scores.values()) / len(scores) for m in metrics}
    means["score"] = means.get(metric, 0.0)
    means["num_topics"] = len(scores)
    return means


def macro_over_domains(per_domain: dict[str, dict[str, float]],
                       metric: str = "score") -> dict[str, float]:
    """Equal-weight mean across domains — **the organizers' baseline-table number**.

    This is what the Phase 1/2 gates reproduce, and what `official_baseline.py` prints. It is
    *not* the leaderboard metric; see `pooled_over_topics` and the module docstring.

    Every domain counts once regardless of how many topics it has. Domains that scored zero
    topics are excluded from the mean rather than counted as 0.0 — a domain we failed to run
    is a bug to fix, not a score to average in. `num_topics` is summed for reporting only and
    plays no part in the weighting.
    """
    usable = {d: v for d, v in per_domain.items() if v.get("num_topics", 0) > 0}
    if not usable:
        return {"macro": 0.0, "num_domains": 0, "num_topics": 0}
    values = [v.get(metric, 0.0) for v in usable.values()]
    return {
        "macro": sum(values) / len(values),
        "num_domains": len(usable),
        "num_topics": sum(int(v.get("num_topics", 0)) for v in usable.values()),
        "skipped_domains": sorted(set(per_domain) - set(usable)),
    }


def pooled_over_topics(per_domain_topic_scores: dict[str, dict[str, object]],
                       metric: str = OFFICIAL_METRIC) -> float:
    """Mean over *all* topics globally, ignoring domain boundaries.

    **This is the leaderboard metric** (CLAUDE.md §4) — 1a nDCG@10 "computed independently for
    each query and macro-averaged". Large domains dominate it by design: History is ~46% of
    train+dev. Optimize this one.

    Caveat for 1b: the official 1b aggregation averages a query's steps first and *then* across
    queries, so a query with 8 steps counts once, not eight times. Passing a flat table of step
    scores here computes the step-level mean instead, which is a different number. Group by
    parent query first when scoring 1b.

    Accepts either shape, because both occur in this codebase and confusing them silently
    would be worse than accepting both: ``{domain: {topic: {metric: value}}}`` as returned by
    `per_topic_scores`, or the flat ``{domain: {topic: value}}`` used by `eval.bootstrap` and
    `eval.cv`.
    """
    all_scores: list[float] = []
    for topics in per_domain_topic_scores.values():
        for entry in topics.values():
            if isinstance(entry, dict):
                all_scores.append(float(entry.get(metric, 0.0)))
            else:
                all_scores.append(float(entry))
    return sum(all_scores) / len(all_scores) if all_scores else 0.0


def render(per_domain: dict[str, dict[str, float]], macro: dict[str, float],
           metric_name: str = "nDCG@10") -> str:
    """Per-domain table plus the macro line. Per CLAUDE.md §6, always report per-domain."""
    lines = [f"  {'domain':<14} {metric_name:>9} {'topics':>7}",
             f"  {'-' * 14} {'-' * 9:>9} {'-' * 7:>7}"]
    for domain, vals in sorted(per_domain.items()):
        lines.append(f"  {domain:<14} {vals.get('score', 0.0):9.4f} "
                     f"{int(vals.get('num_topics', 0)):7d}")
    lines += [
        f"  {'-' * 14} {'-' * 9:>9} {'-' * 7:>7}",
        f"  {'MACRO':<14} {macro['macro']:9.4f} {macro['num_topics']:7d}"
        f"   ({macro['num_domains']} domains, equal weight)",
    ]
    if macro.get("skipped_domains"):
        lines.append(f"  skipped (0 topics): {', '.join(macro['skipped_domains'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="single TREC run file")
    parser.add_argument("--qrels", type=Path, help="qrels for --run")
    parser.add_argument("--domain", default="unknown", help="domain label for --run")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    if not (args.run and args.qrels):
        parser.error("--run and --qrels are both required")

    run = read_run_with_scores(args.run)
    qrels = load_qrels(args.qrels)
    per_domain = {args.domain: domain_score(run, qrels)}
    macro = macro_over_domains(per_domain)

    if args.json:
        print(json.dumps({"per_domain": per_domain, "macro": macro}, indent=2, default=str))
    else:
        print(render(per_domain, macro))
        print("\n  Single domain: the macro equals the domain score. The official number is a\n"
              "  macro over all 13 domains (CLAUDE.md §4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
