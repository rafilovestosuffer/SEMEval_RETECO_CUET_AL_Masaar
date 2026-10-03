#!/usr/bin/env python3
"""Move TREC runs made on the v1.0 corpus onto the v1.1 (deduplicated) ids.

The organizers' v1.1 release (2026-09-24) keeps one copy of each duplicated text and ships
``track1_tempo/<domain>/duplicate_map.json`` (removed id -> kept id). Their instruction for
scoring a v1.0 run: replace every id through the map and drop repeats within a ranking.

Usage::

    python eval/remap_runs.py --runs cache/pipeline_dev/final --maps cache/v11/track1_tempo \
        --out cache/pipeline_dev/final_v11

``--runs`` holds ``<domain>/run_*.trec``; ``--maps`` holds ``<domain>/duplicate_map.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reteco.runs import read_run_ranked, remap_run, write_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    for domain_dir in sorted(p for p in args.runs.iterdir() if p.is_dir()):
        map_path = args.maps / domain_dir.name / "duplicate_map.json"
        if not map_path.is_file():
            print(f"  {domain_dir.name}: no duplicate_map.json at {map_path}", file=sys.stderr)
            return 2
        dup = json.loads(map_path.read_text(encoding="utf-8"))
        for run_path in sorted(domain_dir.glob("run_*.trec")):
            ranked = read_run_ranked(run_path)
            write_run(args.out / domain_dir.name / run_path.name, remap_run(ranked, dup))
            total += 1
    print(f"remapped {total} run files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
