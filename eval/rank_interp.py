#!/usr/bin/env python3
"""v2 step 1 — interpolate the reranked and fused 1a rankings by rank (train only).

ReasonRank outputs an order, not calibrated scores, so the blend is rank-based:
``score(d) = a / (k + rank_reranked(d)) + (1 - a) / (k + rank_fused(d))``. ``a = 1`` is v1
(reranker order alone). Only topics present in the reranked run are scored, and the
comparison is paired against v1 on exactly those topics. Dev is not touched.

Usage::

    python eval/rank_interp.py --reranked cache/p7/runs --fused cache/p5fused/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import paired_bootstrap  # noqa: E402
from reteco.data import load_qrels  # noqa: E402
from reteco.runs import read_run  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402

ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
KS = (10, 30, 60)


def interpolate(reranked: list[str], fused: list[str], a: float, k: int) -> dict[str, float]:
    rr = {d: i + 1 for i, d in enumerate(reranked)}
    fu = {d: i + 1 for i, d in enumerate(fused)}
    miss = len(fused) + 1
    return {d: a / (k + rr.get(d, miss)) + (1 - a) / (k + fu.get(d, miss))
            for d in set(rr) | set(fu)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--qrels-dir", type=Path, default=Path("cache/qrels/track1_tempo"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    results: dict[tuple[float, int], dict[str, dict[str, float]]] = {}
    for dom in sorted(p.name for p in args.reranked.iterdir() if p.is_dir()):
        rr = read_run(args.reranked / dom / f"run_1a_{args.split}.trec")
        fu = read_run(args.fused / dom / f"run_1a_{args.split}.trec")
        qrels = load_qrels(args.qrels_dir / dom / f"qrels_{args.split}.txt")
        for a in ALPHAS:
            for k in KS:
                run = {t: interpolate(rr[t], fu[t], a, k) for t in rr}
                s = per_topic_scores(run, qrels)
                results.setdefault((a, k), {})[dom] = {t: v[OFFICIAL_METRIC] for t, v in s.items()}

    base = results[(1.0, KS[0])]
    flat = lambda r: [x for v in r.values() for x in v.values()]  # noqa: E731
    print(f"v1 (reranker order alone): {sum(flat(base)) / len(flat(base)):.4f} "
          f"over {len(flat(base))} topics\n")
    for (a, k), r in sorted(results.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        if a == 1.0 and k != KS[0]:
            continue
        p = paired_bootstrap(r, base, iters=10000)
        print(f"  a={a:.1f} k={k:<3} {sum(flat(r)) / len(flat(r)):.4f}  {p['delta']:+.4f} "
              f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] {'SIG' if p['significant'] else 'ns'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
