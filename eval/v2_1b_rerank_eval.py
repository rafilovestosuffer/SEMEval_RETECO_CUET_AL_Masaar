#!/usr/bin/env python3
"""v2 step 2 — does ReasonRank-7B over the dense 1b step lists help? (train only, §5.2)

Compares, on exactly the steps the rerank kernel processed (all steps of its sampled
parents), the plain dense step run against the reranked run. Both are mapped onto the v1.1
corpus ids and scored against the v1.1 step qrels with the official 1b aggregation (steps
averaged per query, then across queries), paired bootstrap over queries.

Also reports **sibling promotion**, the risk the v2 plan named: top-10 documents that are
gold for the parent query (1a qrels) but not for the step being ranked.

Usage::

    python eval/v2_1b_rerank_eval.py --dense cache/v2_input/runs --reranked cache/v2_rerank_out/runs \
        --data cache/v11/track1_tempo --out cache/v2_rerank_out/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import paired_bootstrap  # noqa: E402
from reteco.data import load_qrels  # noqa: E402
from reteco.runs import read_run_ranked, remap_run  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402
from score_runs import group_steps_by_query, parent_query_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="v1.1 track1_tempo")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base: dict[str, dict[str, float]] = {}
    rer: dict[str, dict[str, float]] = {}
    promo = {"dense": 0, "reranked": 0, "slots": 0}
    n_steps = 0
    for d in sorted(p.name for p in args.reranked.iterdir() if p.is_dir()):
        dup = json.loads((args.data / d / "duplicate_map.json").read_text("utf-8"))
        rr = remap_run(read_run_ranked(args.reranked / d / f"run_1b_{args.split}.trec"), dup)
        dn = remap_run(read_run_ranked(args.dense / d / f"run_1b_{args.split}.trec"), dup)
        dn = {t: dn[t] for t in rr}                      # same steps only
        qrels = load_qrels(args.data / d / f"qrels_steps_{args.split}.txt")
        parent_qrels = load_qrels(args.data / d / f"qrels_{args.split}.txt")
        for name, run, store in (("dense", dn, base), ("reranked", rr, rer)):
            scored = per_topic_scores({t: dict(v) for t, v in run.items()}, qrels)
            store[d] = group_steps_by_query({t: v[OFFICIAL_METRIC] for t, v in scored.items()})
            for t, docs in run.items():
                step_gold = {x for x, g in qrels.get(t, {}).items() if g > 0}
                par_gold = {x for x, g in parent_qrels.get(parent_query_id(t), {}).items() if g > 0}
                top = [x for x, _ in docs[:10]]
                promo[name] += sum(1 for x in top if x in par_gold and x not in step_gold)
                if name == "dense":
                    promo["slots"] += len(top)
        n_steps += len(rr)
        print(f"  {d:<12} {len(rr):>4} steps", flush=True)

    flat = lambda r: [x for v in r.values() for x in v.values()]  # noqa: E731
    b, r = flat(base), flat(rer)
    p = paired_bootstrap(rer, base, iters=3000)
    result = {"steps": n_steps, "queries": len(b), "dense": sum(b) / len(b),
              "reranked": sum(r) / len(r), "paired": p, "sibling_promotion_top10": promo,
              "per_domain": {d: {"n": len(base[d]), "dense": sum(base[d].values()) / len(base[d]),
                                 "reranked": sum(rer[d].values()) / len(rer[d])} for d in base}}
    print(f"\n  {n_steps} steps, {len(b)} queries (official 1b aggregation, v1.1 ids/qrels)")
    print(f"  dense    {result['dense']:.4f}\n  reranked {result['reranked']:.4f}   "
          f"delta {p['delta']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] "
          f"{'SIGNIFICANT' if p['significant'] else 'ns'}")
    print(f"  sibling-gold docs in top-10: dense {promo['dense']}, reranked {promo['reranked']} "
          f"(of {promo['slots']} slots)")
    for d, v in sorted(result["per_domain"].items()):
        print(f"    {d:<12} n={v['n']:>3} {v['dense']:.4f} -> {v['reranked']:.4f}")
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
