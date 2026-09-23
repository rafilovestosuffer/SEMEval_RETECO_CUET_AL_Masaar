#!/usr/bin/env python3
"""Phase 5c — does the parent query's pool help sub-track 1b, the mirror of H2?

Phase 5 found that 1a gains from the *union* of the whole-query pool with the step pools,
and that a parent weight of 0.001 captures the whole gain (pool membership, not blending).
The mirror question for 1b: fuse each step's own ranking with its parent query's 1a ranking
and score against the **step** qrels, aggregated the official way (steps averaged per query,
then across queries). Train split only; dev is sacred (§5.2).

``w`` is the parent's weight against the step's weight 1, with the same operator,
normalisation and corpus-order tie-break as the frozen 1a config.

Usage::

    python eval/stepfuse_1b.py --runs cache/p4cruns32/runs --out cache/stepfuse_1b
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
from reteco.runs import rank_documents  # noqa: E402
from reteco.stepfuse import fuse_query_and_steps, parent_of  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402
from score_runs import group_steps_by_query  # noqa: E402
from stepfuse_sweep import load_domain_runs  # noqa: E402

WEIGHTS = (0.0, 0.001, 0.1, 0.25, 0.5, 1.0)


def fuse_steps_with_parent(query_run, step_run, doc_order, w: float, top_k: int = 100):
    position = {d: i for i, d in enumerate(doc_order)}
    out = {}
    for sid, ranked in step_run.items():
        parent = query_run.get(parent_of(sid), [])
        fused = fuse_query_and_steps(list(parent), [list(ranked)], parent_weight=w,
                                     operator="sum", normalise="theoretical")
        items = sorted(fused.items(), key=lambda kv: position.get(kv[0], len(position)))
        out[sid] = dict(rank_documents([d for d, _ in items], [s for _, s in items], top_k))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels-dir", type=Path, default=Path("cache/qrels/track1_tempo"))
    parser.add_argument("--doc-index", type=Path, default=Path("cache/docidx/embeddings"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("cache/stepfuse_1b"))
    args = parser.parse_args()

    results: dict[float, dict[str, dict[str, float]]] = {w: {} for w in WEIGHTS}
    for d in sorted(p.name for p in args.runs.iterdir() if p.is_dir()):
        qrels_path = args.qrels_dir / d / f"qrels_steps_{args.split}.txt"
        if not qrels_path.is_file():
            continue
        query_run, step_run = load_domain_runs(args.runs, d, args.split)
        qrels = load_qrels(qrels_path)
        doc_order = json.loads((args.doc_index / d / "doc_index.json")
                               .read_text(encoding="utf-8"))["doc_ids"]
        for w in WEIGHTS:
            fused = fuse_steps_with_parent(query_run, step_run, doc_order, w)
            per_step = {t: v[OFFICIAL_METRIC] for t, v in per_topic_scores(fused, qrels).items()}
            results[w][d] = group_steps_by_query(per_step)
        print(f"  {d:<12} {len(step_run):>5} steps", flush=True)

    base = results[0.0]
    flat = lambda r: [x for v in r.values() for x in v.values()]  # noqa: E731
    print(f"\n  w=0 is the plain 1b step run: {sum(flat(base)) / len(flat(base)):.4f} "
          f"over {len(flat(base))} queries (official 1b aggregation)\n")
    args.out.mkdir(parents=True, exist_ok=True)
    for w in WEIGHTS:
        vals = flat(results[w])
        p = paired_bootstrap(results[w], base, iters=3000)
        verdict = "SIGNIFICANT" if p["significant"] else "ns"
        print(f"  parent w={w:<6g} {sum(vals) / len(vals):.4f}  {p['delta']:+.4f} "
              f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] {verdict}")
        (args.out / f"w{w:g}.json").write_text(json.dumps(results[w]), encoding="utf-8")
    print(f"\n  per-query scores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
