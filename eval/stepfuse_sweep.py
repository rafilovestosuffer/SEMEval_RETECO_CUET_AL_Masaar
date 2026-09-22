#!/usr/bin/env python3
"""Phase 5 — sweep the parent-versus-steps weight for step fusion (H2).

Fuses each query's 1b step rankings into a 1a ranking and scores the result against the
**1a** qrels, so the output is directly comparable with whole-query retrieval. Train split
only; dev is sacred (§5.2).

The sweep is the experiment, not a hyperparameter search to report a best-of. Its endpoints:

- ``w -> large`` drives the fused ranking toward whole-query retrieval. It does **not**
  reproduce it exactly, and the reason is worth knowing: fp16 score quantisation leaves
  ~70% of a top-100 in exact-score tie groups, so an infinitesimal step contribution still
  reorders tied parent documents. Compare against the separately scored 1a run
  (``--baseline``) rather than trusting the high-``w`` column as the baseline.
- ``w = 0`` drops the whole-query list and fuses only the step lists. **This is not TEMPO's
  Step-Only condition**, and an earlier version of this file wrongly called it that. TEMPO's
  Step-Only retrieves with the step text alone (14.6 against Query+Step 26.4); our 1b runs
  already use the official Query+Step template, so every step list here contains the full
  query. ``w = 0`` therefore removes redundancy, not context, and should be expected to do
  far better than 14.6 would suggest.

What is actually novel here is the middle of that curve. No paper reviewed in
`reports/Temporal query and fusion design.md` sweeps the parent-versus-children weight —
they either replace the query or fuse with implicit equal weight.

Usage::

    python eval/stepfuse_sweep.py --runs cache/p4cruns/runs --out cache/stepfuse
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
from reteco.runs import read_run_with_scores  # noqa: E402
from reteco.stepfuse import fuse_run  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402

__all__ = ["DEFAULT_WEIGHTS", "load_domain_runs", "sweep"]

# Geometric-ish ladder: dense near the equal-weight region where the optimum is expected,
# sparse at the ends where the curve should flatten into the two known anchors.
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e6)


def load_domain_runs(runs_root: Path, domain: str, split: str):
    """The 1a run, the 1b step run, and the corpus order for this domain."""
    d = Path(runs_root) / domain
    query_run = read_run_with_scores(d / f"run_1a_{split}.trec")
    step_run = read_run_with_scores(d / f"run_1b_{split}.trec")
    return (
        {q: sorted(v.items(), key=lambda kv: -kv[1]) for q, v in query_run.items()},
        {s: sorted(v.items(), key=lambda kv: -kv[1]) for s, v in step_run.items()},
    )


def sweep(runs_root: Path, qrels_dir: Path, doc_index_dir: Path, split: str,
          weights=DEFAULT_WEIGHTS, operators=("sum",), normalise: str = "theoretical",
          top_k: int = 100) -> dict:
    """Score every (operator, weight) against the 1a qrels, per topic."""
    domains = sorted(p.name for p in Path(runs_root).iterdir() if p.is_dir())
    results: dict[str, dict[str, dict[str, float]]] = {}

    for domain in domains:
        qrels_path = Path(qrels_dir) / domain / f"qrels_{split}.txt"
        if not qrels_path.is_file():
            continue
        query_run, step_run = load_domain_runs(runs_root, domain, split)
        qrels = load_qrels(qrels_path)

        index_path = Path(doc_index_dir) / domain / "doc_index.json"
        doc_order = None
        if index_path.is_file():
            doc_order = json.loads(index_path.read_text(encoding="utf-8"))["doc_ids"]

        for operator in operators:
            for w in weights:
                fused = fuse_run(query_run, step_run, doc_order=doc_order,
                                 parent_weight=w, operator=operator,
                                 normalise=normalise, top_k=top_k)
                as_scores = {q: dict(v) for q, v in fused.items()}
                topic_scores = per_topic_scores(as_scores, qrels)
                key = f"{operator}@w={w:g}"
                results.setdefault(key, {})[domain] = {
                    t: v[OFFICIAL_METRIC] for t, v in topic_scores.items()
                }
        print(f"  {domain:<12} {len(query_run):>5} queries, "
              f"{len(step_run):>5} steps", flush=True)
    return results


def query_macro(per_domain: dict[str, dict[str, float]]) -> float:
    flat = [v for t in per_domain.values() for v in t.values()]
    return sum(flat) / len(flat) if flat else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels-dir", type=Path, default=Path("cache/qrels/track1_tempo"))
    parser.add_argument("--doc-index", type=Path,
                        default=Path("cache/docidx/embeddings"),
                        help="holds <domain>/doc_index.json for the corpus-order tie-break")
    parser.add_argument("--split", default="train")
    parser.add_argument("--weights", nargs="+", type=float, default=list(DEFAULT_WEIGHTS),
                        help="parent weights to sweep")
    parser.add_argument("--operators", nargs="+", default=["sum"],
                        choices=("sum", "max", "rrf"))
    parser.add_argument("--normalise", default="theoretical",
                        choices=("theoretical", "minmax", "rank", "none"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="per-topic scores of the plain 1a run; the honest comparison "
                             "point, since high-w fusion does not reproduce it exactly")
    parser.add_argument("--out", type=Path, default=Path("cache/stepfuse"))
    args = parser.parse_args()

    print(f"sweeping {args.operators} x {len(DEFAULT_WEIGHTS)} weights, "
          f"normalise={args.normalise}\n")
    results = sweep(args.runs, args.qrels_dir, args.doc_index, args.split,
                    weights=tuple(args.weights),
                    operators=tuple(args.operators), normalise=args.normalise)
    if not results:
        print("nothing scored", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for key, per_domain in results.items():
        (args.out / f"{key.replace('@', '_').replace('=', '')}.json").write_text(
            json.dumps(per_domain, indent=0), encoding="utf-8")

    anchor = max(results, key=lambda k: float(k.split("=")[1]))
    if args.baseline and args.baseline.is_file():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        label = f"the 1a run ({args.baseline.name})"
    else:
        base = results[anchor]
        label = f"high-w anchor {anchor} — APPROXIMATE, see the module docstring"
    print(f"\n  baseline: {label} -> {query_macro(base):.4f}")
    print(f"  high-w anchor {anchor} -> {query_macro(results[anchor]):.4f}"
          f"  (differs from the 1a run: step scores break fp16 ties)\n")
    print(f"  {'config':<18}{'query macro':>13}{'delta':>10}{'paired 95% CI':>24}  verdict")
    print(f"  {'-'*18}{'-'*13:>13}{'-'*10:>10}{'-'*24:>24}  -------")
    for key in sorted(results, key=lambda k: (k.split('@')[0], float(k.split('=')[1]))):
        scores = results[key]
        macro = query_macro(scores)

        paired = paired_bootstrap(scores, base, iters=3000)
        verdict = "SIGNIFICANT" if paired["significant"] else "ns"
        print(f"  {key:<18}{macro:>13.4f}{paired['delta']:>+10.4f}"
              f"  [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]  {verdict}")

    print("\n  w=0 drops the whole-query list, but each step list already contains the full")
    print("  query (the official Query+Step template), so this removes redundancy, not")
    print("  context. It is NOT TEMPO's Step-Only condition and must not be read against")
    print("  its 14.6 — an earlier version of this script made exactly that mistake.")
    print(f"\n  per-topic scores -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
