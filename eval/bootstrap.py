#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired comparison for macro nDCG@10.

CLAUDE.md §9 Phase 3 wants "bootstrap 95% CI, paired comparison vs. previous best", and §5
forbids reporting a gain that is inside the noise. Two things make this non-obvious here:

1. **The statistic is the two-level macro**, so a resample has to respect domain structure.
   Resampling topics globally would let a big domain swamp the interval, which is exactly the
   weighting the official metric rejects. `bootstrap_macro` resamples topics *within* each
   domain, recomputes each domain mean, then takes the equal-weight macro.
2. **Comparisons must be paired.** Two systems are scored on the same topics, so the paired
   difference has far lower variance than the two intervals do. Overlapping CIs do **not**
   imply no significant difference — use `paired_bootstrap`, not eyeballed error bars.

With ~10 topics in a domain like IOTA the interval will be very wide, and that is the honest
answer (§6). A per-domain CI is reported alongside the macro so a tiny domain cannot quietly
drive a conclusion.

Seeds are fixed (§11): the same inputs always give the same interval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

__all__ = ["DEFAULT_SEED", "DEFAULT_ITERS", "bootstrap_macro", "paired_bootstrap",
           "percentile_ci"]

DEFAULT_SEED = 20260916
DEFAULT_ITERS = 10_000
DEFAULT_ALPHA = 0.05

# {domain: {topic_id: score}}
PerDomainTopics = dict[str, dict[str, float]]


def _macro(per_domain: dict[str, list[float]]) -> float:
    """Equal-weight mean of per-domain means. Domains with no topics are skipped."""
    means = [sum(v) / len(v) for v in per_domain.values() if v]
    return sum(means) / len(means) if means else 0.0


def percentile_ci(values: list[float], alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Percentile interval at level ``1 - alpha``."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    lo = ordered[max(0, int((alpha / 2) * len(ordered)))]
    hi = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return (lo, hi)


def bootstrap_macro(scores: PerDomainTopics, iters: int = DEFAULT_ITERS,
                    alpha: float = DEFAULT_ALPHA,
                    seed: int = DEFAULT_SEED) -> dict[str, object]:
    """CI for the macro, resampling topics **within** each domain.

    Args:
        scores: ``{domain: {topic_id: per-topic metric}}`` — e.g. ndcg_cut_10 per topic.

    Returns:
        point estimate, CI bounds, and a per-domain CI table.
    """
    rng = random.Random(seed)
    by_domain = {d: list(t.values()) for d, t in scores.items() if t}
    if not by_domain:
        return {"macro": 0.0, "ci_low": 0.0, "ci_high": 0.0, "iters": 0, "per_domain": {}}

    point = _macro(by_domain)

    macro_samples: list[float] = []
    domain_samples: dict[str, list[float]] = {d: [] for d in by_domain}
    for _ in range(iters):
        resampled = {
            d: [values[rng.randrange(len(values))] for _ in values]
            for d, values in by_domain.items()
        }
        for d, values in resampled.items():
            domain_samples[d].append(sum(values) / len(values))
        macro_samples.append(_macro(resampled))

    low, high = percentile_ci(macro_samples, alpha)
    return {
        "macro": point,
        "ci_low": low,
        "ci_high": high,
        "iters": iters,
        "seed": seed,
        "per_domain": {
            d: {
                "score": sum(values) / len(values),
                "num_topics": len(values),
                "ci_low": percentile_ci(domain_samples[d], alpha)[0],
                "ci_high": percentile_ci(domain_samples[d], alpha)[1],
            }
            for d, values in by_domain.items()
        },
    }


def paired_bootstrap(a: PerDomainTopics, b: PerDomainTopics,
                     iters: int = DEFAULT_ITERS, alpha: float = DEFAULT_ALPHA,
                     seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Paired CI for ``macro(a) - macro(b)`` over the topics both systems scored.

    Resamples *topic ids* within each domain and evaluates both systems on the same draw, so
    the shared per-topic difficulty cancels. Only topics present in both are used; a system
    missing topics the other has would otherwise bias the difference.

    ``significant`` is True when the interval excludes 0 — the check CLAUDE.md §9 asks for
    before declaring one configuration better than another.
    """
    rng = random.Random(seed)
    shared: dict[str, list[str]] = {}
    for domain in set(a) & set(b):
        common = sorted(set(a[domain]) & set(b[domain]))
        if common:
            shared[domain] = common
    if not shared:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "significant": False,
                "num_topics": 0}

    def macro_of(system: PerDomainTopics, picks: dict[str, list[str]]) -> float:
        return _macro({d: [system[d][t] for t in ids] for d, ids in picks.items()})

    point = macro_of(a, shared) - macro_of(b, shared)

    deltas = []
    for _ in range(iters):
        picks = {d: [ids[rng.randrange(len(ids))] for _ in ids] for d, ids in shared.items()}
        deltas.append(macro_of(a, picks) - macro_of(b, picks))

    low, high = percentile_ci(deltas, alpha)
    return {
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "significant": (low > 0.0) or (high < 0.0),
        "num_topics": sum(len(ids) for ids in shared.values()),
        "num_domains": len(shared),
        "iters": iters,
        "seed": seed,
    }


def render(result: dict[str, object]) -> str:
    """Macro with CI, then the per-domain table (§6: always report per-domain)."""
    lines = [
        f"macro {result['macro']:.4f}  95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        f"   ({result['iters']} resamples, seed {result['seed']})",
        "",
        f"  {'domain':<14} {'score':>8} {'ci_low':>8} {'ci_high':>8} {'topics':>7}",
        f"  {'-' * 14} {'-' * 8:>8} {'-' * 8:>8} {'-' * 8:>8} {'-' * 7:>7}",
    ]
    for domain, vals in sorted(result["per_domain"].items()):  # type: ignore[union-attr]
        lines.append(f"  {domain:<14} {vals['score']:8.4f} {vals['ci_low']:8.4f} "
                     f"{vals['ci_high']:8.4f} {vals['num_topics']:7d}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True,
                        help='JSON {"domain": {"topic": score}}')
    parser.add_argument("--compare", type=Path, help="second scores file for a paired test")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    if args.compare:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        result = paired_bootstrap(scores, other, args.iters, seed=args.seed)
        verdict = "SIGNIFICANT" if result["significant"] else "not significant"
        print(f"delta {result['delta']:+.4f}  95% CI "
              f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  -> {verdict}")
        print(f"paired over {result['num_topics']} topics in {result['num_domains']} domains")
    else:
        print(render(bootstrap_macro(scores, args.iters, seed=args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
