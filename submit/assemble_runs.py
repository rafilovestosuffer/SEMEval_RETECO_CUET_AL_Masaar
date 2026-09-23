#!/usr/bin/env python3
"""Frozen pipeline, final local step — assemble and validate the runs that get submitted.

Inputs are the outputs of the three pipeline stages for one split:

- ``--fused``   ``<domain>/run_1a_<split>.trec`` (fused 1a) and ``run_1b_<split>.trec`` (dense
  1b), from ``eval/write_fused_runs.py``;
- ``--reranked`` ``<domain>/run_1a_<split>.trec`` from ``kaggle/kernels/pipeline_rerank``
  (optional; omit it to submit the fused system).

For every 1a topic the reranked list is used when present and the fused list otherwise, so a
reranking session cut short by the time budget degrades to the fused system on the topics
it did not reach rather than dropping them — a missing topic scores 0 (CLAUDE.md §4).

Then, per domain and sub-track, it checks coverage against the split's own query files
(``examples_<split>.jsonl`` / ``steps_<split>.jsonl``: every topic must have a row) and runs
the organizers' ``format_checker.py`` through ``submit/check_format.py``. Any failure is a
non-zero exit. qrels are never read to build a run, only optionally to validate topic ids.

Usage::

    python submit/assemble_runs.py --split dev --data cache/pipeline_dev/data/track1_tempo \\
        --fused cache/pipeline_dev/fused --reranked cache/pipeline_dev/reranked \\
        --out cache/pipeline_dev/final
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_format import validate  # noqa: E402
from reteco.data import read_jsonl  # noqa: E402
from reteco.runs import read_run_with_scores, write_run  # noqa: E402

__all__ = ["merge_1a", "expected_topics", "main"]


def merge_1a(fused: dict[str, dict[str, float]], reranked: dict[str, dict[str, float]]
             ) -> tuple[dict[str, list[tuple[str, float]]], int]:
    """Reranked list where available, fused otherwise. Returns (run, topics reranked)."""
    out: dict[str, list[tuple[str, float]]] = {}
    used = 0
    for topic in fused:
        source = reranked.get(topic)
        if source:
            used += 1
        else:
            source = fused[topic]
        out[topic] = sorted(source.items(), key=lambda kv: kv[1], reverse=True)
    return out, used


def expected_topics(domain_dir: Path, split: str) -> dict[str, set[str]]:
    """Topic ids the split defines, read from its query files (never from qrels)."""
    topics = {"1a": set(), "1b": set()}
    ex = domain_dir / f"examples_{split}.jsonl"
    if ex.is_file():
        topics["1a"] = {r["id"] for r in read_jsonl(ex)}
    st = domain_dir / f"steps_{split}.jsonl"
    if st.is_file():
        topics["1b"] = {s["step_id"] for r in read_jsonl(st) for s in r.get("steps", [])}
    return topics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="track1_tempo root with <domain>/examples_<split>.jsonl, "
                             "steps_<split>.jsonl and documents.jsonl")
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-corpus-check", action="store_true",
                        help="skip passing documents.jsonl to the checker (faster)")
    args = parser.parse_args()

    failures: list[str] = []
    domains = sorted(p.name for p in args.fused.iterdir() if p.is_dir())
    total = {"1a": 0, "1b": 0, "reranked": 0}
    for domain in domains:
        src, dst = args.fused / domain, args.out / domain
        want = expected_topics(args.data / domain, args.split)

        fused_1a = read_run_with_scores(src / f"run_1a_{args.split}.trec")
        reranked = {}
        if args.reranked is not None:
            path = args.reranked / domain / f"run_1a_{args.split}.trec"
            if path.is_file():
                reranked = read_run_with_scores(path)
        run_1a, used = merge_1a(fused_1a, reranked)
        write_run(dst / f"run_1a_{args.split}.trec", run_1a, tag="cuet_al_masaar")
        shutil.copyfile(src / f"run_1b_{args.split}.trec", dst / f"run_1b_{args.split}.trec")
        total["reranked"] += used

        for sub in ("1a", "1b"):
            run_path = dst / f"run_{sub}_{args.split}.trec"
            have = set(read_run_with_scores(run_path))
            missing = want[sub] - have
            extra = have - want[sub]
            total[sub] += len(have)
            if missing:
                failures.append(f"{domain} {sub}: {len(missing)} topics have no rows, "
                                f"e.g. {sorted(missing)[:3]}")
            if extra:
                failures.append(f"{domain} {sub}: {len(extra)} topics not in the split files")
            corpus = None if args.no_corpus_check else args.data / domain / "documents.jsonl"
            result = validate(run_path, corpus=corpus if corpus and corpus.is_file() else None)
            if not result.valid:
                failures.append(f"{domain} {sub}: checker INVALID — {result.errors[:3]}")
        print(f"  {domain:<12} 1a {len(run_1a):>4} topics ({used} reranked), "
              f"1b {len(want['1b']):>4} steps", flush=True)

    print(f"\n  1a {total['1a']} topics ({total['reranked']} reranked), 1b {total['1b']} steps"
          f"\n  runs -> {args.out}")
    if failures:
        print("\n  NOT SUBMITTABLE:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  every topic covered, every file VALID under the organizers' checker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
