#!/usr/bin/env python3
"""Official scoring: `pytrec_eval` nDCG@10, macro-averaged the way RETECO does it.

**The averaging is two-level and getting it wrong is the easiest way to report a number that
looks plausible and is not ours** (CLAUDE.md §4, §6):

1. within a domain, `pytrec_eval` scores each topic and we take the **mean over topics**;
2. across domains, we take the **equal-weight mean over the 13 domains**.

Pooling all topics globally gives a different answer, because domain sizes differ by two orders
of magnitude — History has 801 queries, IOTA about 10. Verified against the organizers' table:
the 13 published per-domain 1a-train values average to 0.08785 → their published 0.0879.

`macro_over_domains` therefore takes per-domain results, never a flat topic list. The
`pooled_over_topics` function exists only so the difference can be reported in the paper; it is
**not** the official metric and says so.

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
    """Level 2: equal-weight mean across domains. **This is the official number.**

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

    **NOT the official metric.** Provided only so the gap between pooled and macro averaging
    can be quantified in the paper — large domains dominate it, which is precisely why RETECO
    does not use it. Never report this as our score.

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
