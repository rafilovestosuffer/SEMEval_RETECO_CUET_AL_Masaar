#!/usr/bin/env python3
"""Train-only cross-validation, stratified by domain (CLAUDE.md §9 Phase 3).

**Dev is sacred** (§5.2). Every configuration choice — first-stage model, fusion weights,
rerank depth K — is made on train folds and nothing else. This module exists so there is never
a reason to look at dev while tuning.

Two design points follow directly from the metric being a per-domain macro (§4, §6):

- **Folds are stratified by domain.** Each fold holds roughly the same *proportion* of every
  domain's topics, so every fold can compute all 13 domain means. Unstratified folds would
  leave small domains absent from some folds, and a domain missing from a fold cannot
  contribute to that fold's macro — which silently reweights the metric.
- **Per-domain spread is reported, not just the macro.** With ~10 topics, a domain's fold
  scores swing hugely; a macro that looks stable can hide a domain oscillating between 0.0
  and 0.4.

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

from bootstrap import DEFAULT_SEED, bootstrap_macro, paired_bootstrap  # noqa: E402

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


def _macro(scores: PerDomainTopics) -> float:
    means = [sum(t.values()) / len(t) for t in scores.values() if t]
    return sum(means) / len(means) if means else 0.0


def cross_validate(scores: PerDomainTopics, n_folds: int = DEFAULT_FOLDS,
                   seed: int = DEFAULT_SEED, iters: int = 10_000) -> dict[str, object]:
    """Fold-wise macro, its spread, and a bootstrap CI over all train topics.

    The CI comes from `bootstrap_macro` over the full train set rather than from the fold
    scores: with 5 folds there are only 5 numbers, far too few for a percentile interval. The
    fold spread is reported separately as a stability check — CLAUDE.md §9 asks for "variance
    across folds reported", which is a different question from the CI.
    """
    topics_by_domain = {d: sorted(t) for d, t in scores.items() if t}
    folds = make_folds(topics_by_domain, n_folds, seed)

    per_fold = []
    for i, fold in enumerate(folds):
        subset = fold_scores(scores, fold)
        per_fold.append({
            "fold": i,
            "macro": _macro(subset),
            "num_topics": sum(len(t) for t in subset.values()),
            "num_domains": len(subset),
            "per_domain": {d: sum(t.values()) / len(t) for d, t in sorted(subset.items())},
        })

    macros = [f["macro"] for f in per_fold]
    overall = bootstrap_macro(scores, iters=iters, seed=seed)

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
        "fold_macros": macros,
        "fold_macro_mean": statistics.fmean(macros) if macros else 0.0,
        "fold_macro_stdev": statistics.stdev(macros) if len(macros) > 1 else 0.0,
        "overall_macro": overall["macro"],
        "ci_low": overall["ci_low"],
        "ci_high": overall["ci_high"],
        "per_fold": per_fold,
        "domain_spread": domain_spread,
    }


def render(result: dict) -> str:
    lines = [
        f"{result['n_folds']}-fold CV, stratified by domain (seed {result['seed']})",
        "",
        f"  overall macro {result['overall_macro']:.4f}  "
        f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}]",
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    result = cross_validate(scores, args.folds, args.seed)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render(result))

    if args.compare:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        paired = paired_bootstrap(scores, other, seed=args.seed)
        verdict = "SIGNIFICANT" if paired["significant"] else "not significant"
        print(f"\npaired vs {args.compare.name}: delta {paired['delta']:+.4f}  "
              f"95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]  -> {verdict}")
        print("  Overlapping individual CIs do not imply no difference; this paired test is\n"
              "  the one that decides (§9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
