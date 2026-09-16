#!/usr/bin/env python3
"""End-to-end: raw split files -> validated TREC runs for sub-tracks 1a and 1b.

CLAUDE.md §9 Phase 8 wants this built and dry-run on dev *as if it were test*, before the
January window, so submission day is a re-run rather than a first run.

Design point: **retrieval is a pluggable callable**, so Phase 4's dense retriever drops in
without touching the submission path, and the path itself can be tested today with a trivial
retriever and a synthetic fixture. A retriever is::

    retrieve(corpus: Corpus, topics: list[Topic], top_k: int)
        -> dict[topic_id, list[(doc_id, score)]]

Two guarantees this script enforces, both from CLAUDE.md §4:

- **Every topic gets a row.** Missing topics in an entered sub-track score 0, so an empty
  ranking is a silent loss. Topics that retrieved nothing are reported loudly.
- **Nothing is submitted unvalidated.** Each run goes through the organizers'
  `format_checker.py` (via `submit/check_format.py`) and a failure is a non-zero exit.

It never reads qrels to build a run — only, optionally, to validate topic ids. On the real test
split there will be no qrels at all, so the dry run must work without them (`--no-qrels`).

Usage::

    python submit/make_runs.py --domains iota --split dev --out results/runs --no-qrels
    python submit/make_runs.py --data /path/to/track1_tempo --split test --out submit/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_format import validate  # noqa: E402
from reteco.data import Corpus, Topic, topics_1a, topics_1b, load_corpus  # noqa: E402
from reteco.paths import data_root, ensure_dir, TRACK1_PREFIX  # noqa: E402
from reteco.runs import DEFAULT_TAG, TOP_K, rank_documents, write_run  # noqa: E402

__all__ = ["Retriever", "lexical_overlap_retriever", "make_domain_runs", "main"]

Retriever = Callable[[Corpus, Sequence[Topic], int], dict[str, list[tuple[str, float]]]]
SUBTRACKS = {"1a": topics_1a, "1b": topics_1b}


def lexical_overlap_retriever(corpus: Corpus, topics: Sequence[Topic],
                              top_k: int = TOP_K) -> dict[str, list[tuple[str, float]]]:
    """Trivial token-overlap retriever — a placeholder so the path is testable today.

    **Not a baseline and not competitive.** The official BM25 comes from the organizers'
    `official_baseline.py` (pyserini Lucene + gensim), and Phase 4 replaces this with the
    dense retriever. It exists only so `make_runs.py` can be exercised end to end before any
    real model is wired in.
    """
    doc_tokens = [set(text.lower().split()) for text in corpus.texts]
    out = {}
    for topic in topics:
        query_tokens = set(topic.text.lower().split())
        scores = [float(len(query_tokens & tokens)) for tokens in doc_tokens]
        out[topic.topic_id] = rank_documents(corpus.doc_ids, scores, top_k)
    return out


def make_domain_runs(domain: str, split: str, out_dir: Path, retriever: Retriever,
                     data_dir: Path | None = None, subtracks: Sequence[str] = ("1a", "1b"),
                     top_k: int = TOP_K, tag: str = DEFAULT_TAG,
                     qrels: bool = True) -> dict[str, dict]:
    """Build and validate runs for one domain. Returns a per-sub-track report."""
    domain_path = (Path(data_dir) / domain) if data_dir else None
    corpus = load_corpus(domain, root=domain_path)
    report: dict[str, dict] = {}

    for subtrack in subtracks:
        topics = SUBTRACKS[subtrack](domain, split, domain_path)
        if not topics:
            report[subtrack] = {"skipped": "no topics"}
            continue

        ranked = retriever(corpus, topics, top_k)

        empty = [t.topic_id for t in topics if not ranked.get(t.topic_id)]
        missing = [t.topic_id for t in topics if t.topic_id not in ranked]

        ordered = {t.topic_id: ranked.get(t.topic_id, []) for t in topics}
        run_path = ensure_dir(Path(out_dir)) / f"run_{domain}_{subtrack}_{split}.trec"
        write_run(run_path, ordered, tag=tag)

        qrels_name = "qrels" if subtrack == "1a" else "qrels_steps"
        qrels_path = None
        if qrels and domain_path is not None:
            candidate = domain_path / f"{qrels_name}_{split}.txt"
            qrels_path = candidate if candidate.is_file() else None

        corpus_path = (domain_path / "documents.jsonl") if domain_path else None
        result = validate(run_path, qrels_path, corpus_path)

        report[subtrack] = {
            "run": str(run_path),
            "num_topics": len(topics),
            "topics_with_results": len(topics) - len(empty),
            "empty_topics": empty[:20],
            "missing_topics": missing[:20],
            "valid": result.valid,
            "checker": result.checker,
            "errors": result.errors[:10],
        }

        status = "VALID" if result.valid else "INVALID"
        print(f"  {domain}/{subtrack}: {len(topics)} topics -> {run_path.name}  "
              f"[{result.checker}] {status}")
        if empty:
            print(f"    WARNING: {len(empty)} topic(s) retrieved nothing and will score 0: "
                  f"{', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''}")
        for err in result.errors[:5]:
            print(f"    {err}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None,
                        help="Track 1 root holding the domain dirs "
                             "(default: <RETECO_DATA>/track1_tempo)")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="domains to run (default: every directory under --data)")
    parser.add_argument("--split", default="dev", help="train | dev | test")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <results>/runs)")
    parser.add_argument("--subtracks", nargs="+", default=["1a", "1b"], choices=["1a", "1b"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--no-qrels", action="store_true",
                        help="do not look for qrels — required for the real test split")
    args = parser.parse_args()

    data_dir = args.data or (data_root() / TRACK1_PREFIX)
    if not Path(data_dir).is_dir():
        print(f"no Track 1 data at {data_dir}\n"
              f"Download it first, or point --data at the domain directories.", file=sys.stderr)
        return 1

    domains = args.domains or sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    if not domains:
        print(f"no domain directories under {data_dir}", file=sys.stderr)
        return 1

    out_dir = args.out or (Path(__file__).resolve().parent.parent / "results" / "runs")
    print(f"data   {data_dir}\nsplit  {args.split}\nout    {out_dir}\n"
          f"domains ({len(domains)}): {', '.join(domains)}\n")

    all_valid = True
    for domain in domains:
        report = make_domain_runs(
            domain, args.split, out_dir, lexical_overlap_retriever, Path(data_dir),
            args.subtracks, args.top_k, args.tag, qrels=not args.no_qrels,
        )
        all_valid &= all(r.get("valid", True) for r in report.values())

    print(f"\nruns written to {out_dir}")
    print("RESULT: all runs valid" if all_valid else "RESULT: SOME RUNS INVALID")
    print("\nNOTE: the built-in retriever is a token-overlap placeholder, not a baseline.\n"
          "Phase 4 replaces it; this path exists so submission day is a re-run, not a "
          "first run.")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
