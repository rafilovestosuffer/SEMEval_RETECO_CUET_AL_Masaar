#!/usr/bin/env python3
"""Train-only cross-validation, stratified by domain (CLAUDE.md §9 Phase 3).

**Dev is sacred** (§5.2). Every configuration choice — first-stage model, fusion weights,
rerank depth K — is made on train folds and nothing else. This module exists so there is never
a reason to look at dev while tuning.

Three design points:

- **Both aggregations are reported, and the query macro is primary** (§4, corrected 22 Sept
  2026). It is the one that ranks us; the domain macro is kept because it reproduces the
  organizers' baseline table. On real BM25 train scores they are 0.0944 and 0.0879 — a gap
  wide enough that a change can look like a gain under one and a loss under the other.
- **Folds are stratified by domain.** Each fold holds roughly the same *proportion* of every
  domain's topics, so every fold can compute all 13 domain means. Unstratified folds would
  leave small domains absent from some folds, and a domain missing from a fold cannot
  contribute to that fold's domain macro — which silently reweights it. Stratification is
  right under either aggregation; only the objective changes.
- **Per-domain spread is reported, not just the macro.** With ~10 topics, a domain's fold
  scores swing hugely; a macro that looks stable can hide a domain oscillating between 0.0
  and 0.4. Measured on real data: IOTA has 7 train topics and a fold sd of 0.062 against a
  mean of 0.028, while History has 561 topics and a sd of 0.013.

Folds are deterministic: the same topic ids and seed always produce the same partition, so a
comparison run weeks apart is still paired.

Usage::

    python eval/cv.py --scores scores.json --folds 5
    python eval/cv.py --scores a.json --compare b.json --folds 5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bootstrap import (DEFAULT_MACRO_MODE, DEFAULT_SEED, bootstrap_macro,  # noqa: E402
                       paired_bootstrap)

__all__ = ["DEFAULT_FOLDS", "make_folds", "fold_scores", "cross_validate"]

DEFAULT_FOLDS = 5

PerDomainTopics = dict[str, dict[str, float]]


def make_folds(topics_by_domain: dict[str, list[str]], n_folds: int = DEFAULT_FOLDS,
               seed: int = DEFAULT_SEED) -> list[dict[str, list[str]]]:
    """Partition topics into ``n_folds`` folds, stratified by domain.

    Each domain's topics are shuffled with a fixed seed and dealt round-robin across folds, so
    every fold gets a near-equal share of every domain. Deterministic in the topic ids, not in
    their input order: ids are sorted before shuffling, so a reordered input gives the same
    folds.

    Returns:
        A list of ``{domain: [topic_id, ...]}``, one per fold.
    """
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")

    folds: list[dict[str, list[str]]] = [{} for _ in range(n_folds)]
    for domain in sorted(topics_by_domain):
        topics = sorted(topics_by_domain[domain])
        rng = random.Random(f"{seed}:{domain}")
        rng.shuffle(topics)
        for i, topic_id in enumerate(topics):
            folds[i % n_folds].setdefault(domain, []).append(topic_id)
    return folds


def fold_scores(scores: PerDomainTopics, fold: dict[str, list[str]]) -> PerDomainTopics:
    """Restrict a score table to the topics in one fold."""
    out: PerDomainTopics = {}
    for domain, topic_ids in fold.items():
        available = scores.get(domain, {})
        picked = {t: available[t] for t in topic_ids if t in available}
        if picked:
            out[domain] = picked
    return out


STEP_MARKER = "_step"


def _reject_step_level_ids(topics_by_domain: dict[str, list[str]]) -> None:
    """Refuse to fold sub-track 1b *step* ids — the unit of independence is the query.

    Steps of one query share its phrasing and much of its gold, so splitting them across
    folds leaks between train and held-out exactly as splitting one patient's scans would.
    Measured on the real train split: folding step ids puts **74.8%** of parent queries
    (839 of 1,121) on both sides of a fold boundary.

    `eval/score_runs.py` emits per-query means for 1b by default, which is both the
    official aggregation and the correct fold unit; its ``--by-step`` output is diagnostic
    and must not be fed here. This is a hard error rather than a warning because the
    resulting numbers look entirely plausible.
    """
    offenders = {}
    for domain, ids in topics_by_domain.items():
        parents = {i.rsplit(STEP_MARKER, 1)[0] for i in ids if STEP_MARKER in i}
        steps = sum(1 for i in ids if STEP_MARKER in i)
        if parents and steps > len(parents):
            offenders[domain] = (steps, len(parents))
    if offenders:
        example = ", ".join(f"{d} ({s} steps / {p} queries)"
                            for d, (s, p) in sorted(offenders.items())[:3])
        raise ValueError(
            "these look like sub-track 1b STEP ids, not query ids: " + example +
            ". Steps of one query would be split across folds, which leaks. Score 1b "
            "without --by-step so each query contributes one averaged value."
        )


def _macro(scores: PerDomainTopics, mode: str = DEFAULT_MACRO_MODE) -> float:
    """Fold macro under either aggregation — see `eval.bootstrap._macro`."""
    groups = [list(t.values()) for t in scores.values() if t]
    if not groups:
        return 0.0
    if mode == "query":
        flat = [x for g in groups for x in g]
        return sum(flat) / len(flat)
    means = [sum(g) / len(g) for g in groups]
    return sum(means) / len(means)


def cross_validate(scores: PerDomainTopics, n_folds: int = DEFAULT_FOLDS,
                   seed: int = DEFAULT_SEED, iters: int = 10_000,
                   mode: str = DEFAULT_MACRO_MODE) -> dict[str, object]:
    """Fold-wise macro, its spread, and a bootstrap CI over all train topics.

    The CI comes from `bootstrap_macro` over the full train set rather than from the fold
    scores: with 5 folds there are only 5 numbers, far too few for a percentile interval. The
    fold spread is reported separately as a stability check — CLAUDE.md §9 asks for "variance
    across folds reported", which is a different question from the CI.
    """
    topics_by_domain = {d: sorted(t) for d, t in scores.items() if t}
    _reject_step_level_ids(topics_by_domain)
    folds = make_folds(topics_by_domain, n_folds, seed)

    per_fold = []
    for i, fold in enumerate(folds):
        subset = fold_scores(scores, fold)
        per_fold.append({
            "fold": i,
            "macro": _macro(subset, mode),
            "macro_other": _macro(subset, "domain" if mode == "query" else "query"),
            "num_topics": sum(len(t) for t in subset.values()),
            "num_domains": len(subset),
            "per_domain": {d: sum(t.values()) / len(t) for d, t in sorted(subset.items())},
        })

    macros = [f["macro"] for f in per_fold]
    overall = bootstrap_macro(scores, iters=iters, seed=seed, mode=mode)
    other_mode = "domain" if mode == "query" else "query"
    overall_other = bootstrap_macro(scores, iters=iters, seed=seed, mode=other_mode)

    domains = sorted(topics_by_domain)
    domain_spread = {}
    for domain in domains:
        values = [f["per_domain"][domain] for f in per_fold if domain in f["per_domain"]]
        domain_spread[domain] = {
            "mean": statistics.fmean(values) if values else 0.0,
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "folds_present": len(values),
            "num_topics": len(topics_by_domain[domain]),
        }

    return {
        "n_folds": n_folds,
        "seed": seed,
        "mode": mode,
        "other_mode": other_mode,
        "other_macro": overall_other["macro"],
        "other_ci_low": overall_other["ci_low"],
        "other_ci_high": overall_other["ci_high"],
        "fold_macros": macros,
        "fold_macro_mean": statistics.fmean(macros) if macros else 0.0,
        "fold_macro_stdev": statistics.stdev(macros) if len(macros) > 1 else 0.0,
        "overall_macro": overall["macro"],
        "ci_low": overall["ci_low"],
        "ci_high": overall["ci_high"],
        "per_fold": per_fold,
        "domain_spread": domain_spread,
    }


def _macro_line(mode: str, macro: float, low: float, high: float) -> str:
    """One macro row. The query macro is marked, since it is the one that ranks us (§4)."""
    marker = "  <- optimise this" if mode == "query" else ""
    return (f"  {mode + '-macro':<14} {macro:.4f}  "
            f"95% CI [{low:.4f}, {high:.4f}]{marker}")


def render(result: dict) -> str:
    lines = [
        f"{result['n_folds']}-fold CV, stratified by domain (seed {result['seed']})",
        "",
        _macro_line(result["mode"], result["overall_macro"],
                    result["ci_low"], result["ci_high"]),
        _macro_line(result["other_mode"], result["other_macro"],
                    result["other_ci_low"], result["other_ci_high"]),
        f"  fold macros   {', '.join(f'{m:.4f}' for m in result['fold_macros'])}",
        f"  fold spread   mean {result['fold_macro_mean']:.4f}  "
        f"sd {result['fold_macro_stdev']:.4f}",
        "",
        f"  {'domain':<14} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8} {'topics':>7}",
        f"  {'-' * 14} {'-' * 8:>8} {'-' * 8:>8} {'-' * 8:>8} {'-' * 8:>8} {'-' * 7:>7}",
    ]
    for domain, v in sorted(result["domain_spread"].items()):
        lines.append(f"  {domain:<14} {v['mean']:8.4f} {v['stdev']:8.4f} "
                     f"{v['min']:8.4f} {v['max']:8.4f} {v['num_topics']:7d}")
    lines += [
        "",
        "  Small domains swing hard across folds (§6). A stable macro can hide a domain",
        "  oscillating widely — read the sd column before believing a gain.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True,
                        help='JSON {"domain": {"topic": score}} on the TRAIN split')
    parser.add_argument("--compare", type=Path,
                        help="second scores file; runs a paired bootstrap against it")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", choices=("query", "domain"), default=DEFAULT_MACRO_MODE,
                        help="primary aggregation; 'query' is the leaderboard metric (§4)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    result = cross_validate(scores, args.folds, args.seed, mode=args.mode)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render(result))

    if args.compare:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        paired = paired_bootstrap(scores, other, seed=args.seed, mode=args.mode)
        verdict = "SIGNIFICANT" if paired["significant"] else "not significant"
        print(f"\npaired vs {args.compare.name}: delta {paired['delta']:+.4f}  "
              f"95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]  -> {verdict}")
        print("  Overlapping individual CIs do not imply no difference; this paired test is\n"
              "  the one that decides (§9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
