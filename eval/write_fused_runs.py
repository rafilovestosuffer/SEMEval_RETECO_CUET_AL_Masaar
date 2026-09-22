#!/usr/bin/env python3
"""Write the frozen Phase 5 step-fusion config out as TREC runs.

Phase 5/5b scored fusion in memory; Phase 7 needs the fused 1a ranking *on disk* as the
reranker's candidate list. This writes, per domain:

- ``run_1a_<split>.trec`` — the 1a whole-query run fused with its 1b step runs, using the
  config Phase 5b kept: sum operator, theoretical normalisation, parent_weight 0.001,
  corpus-order tie-break, top-100 (ledger ``phase5_stepfuse_w0.001``, ``phase5b_*``).
- ``run_1b_<split>.trec`` — copied unchanged; 1b is not fused.

Usage::

    python eval/write_fused_runs.py --runs cache/p4cruns32/runs --out cache/p5fused/runs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reteco.runs import write_run  # noqa: E402
from reteco.stepfuse import fuse_run  # noqa: E402
from stepfuse_sweep import load_domain_runs  # noqa: E402

FROZEN = {"parent_weight": 0.001, "operator": "sum", "normalise": "theoretical", "top_k": 100}
TAG = "cuet_stepfuse"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True,
                        help="holds <domain>/run_1a_<split>.trec and run_1b_<split>.trec")
    parser.add_argument("--doc-index", type=Path, default=Path("cache/docidx/embeddings"),
                        help="holds <domain>/doc_index.json for the corpus-order tie-break")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    domains = sorted(p.name for p in args.runs.iterdir() if p.is_dir())
    for domain in domains:
        src = args.runs / domain
        if not (src / f"run_1a_{args.split}.trec").is_file():
            print(f"  skip {domain}: no 1a run")
            continue
        query_run, step_run = load_domain_runs(args.runs, domain, args.split)
        index_path = args.doc_index / domain / "doc_index.json"
        if not index_path.is_file():
            print(f"no doc_index for {domain} at {index_path}; the tie-break needs it",
                  file=sys.stderr)
            return 1
        doc_order = json.loads(index_path.read_text(encoding="utf-8"))["doc_ids"]

        fused = fuse_run(query_run, step_run, doc_order=doc_order, **FROZEN)
        dst = args.out / domain
        write_run(dst / f"run_1a_{args.split}.trec", fused, tag=TAG)
        shutil.copyfile(src / f"run_1b_{args.split}.trec", dst / f"run_1b_{args.split}.trec")
        print(f"  {domain:<12} {len(fused):>5} queries fused, {len(step_run):>5} steps copied")

    print(f"\n  config {FROZEN}\n  runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
