#!/usr/bin/env python3
"""TREC run files + qrels -> the per-topic score table `eval/cv.py` consumes.

This is the missing link between a retrieval run and the Phase 3 validation harness:
`cv.py --scores` wants ``{domain: {topic_id: score}}`` and nothing produced it.

**Sub-track 1b needs care, and getting it wrong inflates the score.** `evaluation.html`:
"step-specific nDCG@10 values are first averaged over the supplied steps for each query,
then aggregated across queries". So the unit that enters the query macro is the *query*,
not the step. A query with 8 steps must count once, not eight times. This script therefore
emits ``{domain: {query_id: mean-over-its-steps}}`` for 1b by default, which makes
`cv.py`'s flat pooling correct without it needing to know about steps. Pass ``--by-step``
to get raw step scores instead — useful for diagnosing which steps fail, never for
reporting the official number.

Step ids are ``<query_id>_step<N>`` (e.g. ``367_805_step1`` -> ``367_805``), verified
against the released qrels on 2026-09-22.

Usage::

    python eval/score_runs.py --subtrack 1a --split train --out cache/scores_1a_train.json
    python eval/score_runs.py --subtrack 1b --split train --by-step --out steps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reteco.data import load_qrels  # noqa: E402
from reteco.runs import read_run_with_scores  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402

__all__ = ["STEP_MARKER", "parent_query_id", "group_steps_by_query", "score_domain"]

STEP_MARKER = "_step"


def parent_query_id(step_id: str) -> str:
    """``367_805_step1`` -> ``367_805``. Returns the id unchanged if it is not a step id."""
    return step_id.rsplit(STEP_MARKER, 1)[0] if STEP_MARKER in step_id else step_id


def group_steps_by_query(step_scores: dict[str, float]) -> dict[str, float]:
    """Average step scores within each parent query — the official 1b aggregation."""
    buckets: dict[str, list[float]] = {}
    for step_id, value in step_scores.items():
        buckets.setdefault(parent_query_id(step_id), []).append(value)
    return {qid: sum(v) / len(v) for qid, v in buckets.items()}


def score_domain(run_path: Path, qrels_path: Path,
                 metric: str = OFFICIAL_METRIC) -> dict[str, float]:
    """Per-topic metric for one domain. Topics absent from either side are skipped."""
    run = read_run_with_scores(run_path)
    qrels = load_qrels(qrels_path)
    return {topic: values[metric] for topic, values in per_topic_scores(run, qrels).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("cache/p2runs/baseline_out/track1_tempo"),
                        help="holds <domain>/run_<subtrack>_<split>.trec")
    parser.add_argument("--qrels-dir", type=Path, default=Path("cache/qrels/track1_tempo"),
                        help="holds <domain>/qrels[_steps]_<split>.txt")
    parser.add_argument("--subtrack", choices=("1a", "1b"), default="1a")
    parser.add_argument("--split", default="train",
                        help="train by default; dev is sacred (§5.2) and must be justified")
    parser.add_argument("--by-step", action="store_true",
                        help="1b only: emit raw step scores instead of the official "
                             "per-query means. Diagnostic, not reportable.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        print(f"no runs directory at {args.runs_dir}", file=sys.stderr)
        return 1

    qrels_name = ("qrels_steps" if args.subtrack == "1b" else "qrels") + f"_{args.split}.txt"
    per_domain: dict[str, dict[str, float]] = {}
    for domain_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        run_path = domain_dir / f"run_{args.subtrack}_{args.split}.trec"
        qrels_path = args.qrels_dir / domain_dir.name / qrels_name
        if not run_path.is_file() or not qrels_path.is_file():
            print(f"  skip {domain_dir.name}: missing run or qrels")
            continue
        scores = score_domain(run_path, qrels_path)
        if args.subtrack == "1b" and not args.by_step:
            scores = group_steps_by_query(scores)
        per_domain[domain_dir.name] = scores

    if not per_domain:
        print("scored nothing", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(per_domain, indent=0), encoding="utf-8")

    units = sum(len(v) for v in per_domain.values())
    flat = [x for v in per_domain.values() for x in v.values()]
    query_macro = sum(flat) / len(flat)
    domain_macro = sum(sum(v.values()) / len(v) for v in per_domain.values()) / len(per_domain)
    unit = "steps" if (args.subtrack == "1b" and args.by_step) else "queries"
    print(f"{args.subtrack}/{args.split}: {len(per_domain)} domains, {units} {unit}")
    print(f"  query-macro  {query_macro:.4f}   <- the leaderboard metric (§4)")
    print(f"  domain-macro {domain_macro:.4f}   (organizers' baseline table)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
