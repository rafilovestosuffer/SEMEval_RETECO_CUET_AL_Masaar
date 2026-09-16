"""TREC run files: ranking, writing, reading.

Run format (CLAUDE.md §4), six whitespace-separated columns::

    topic  Q0  doc_id  rank  score  tag

Topic ids are `id` for sub-track 1a and `step_id` for 1b.

The one subtlety worth a module of its own is **tie-breaking**. The organizers rank with::

    sorted(zip(doc_ids, sims), key=lambda x: x[1], reverse=True)

Python's sort is stable, so documents with identical scores keep their **corpus order** —
their position in `documents.jsonl`. A reimplementation that breaks ties by doc id produces a
different ranking whenever scores tie, which BM25 does often on short queries. `rank_documents`
reproduces the official behaviour exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["rank_documents", "write_run", "read_run", "run_from_scores", "DEFAULT_TAG", "TOP_K"]

DEFAULT_TAG = "cuet_al_masaar"
TOP_K = 1000  # the organizers score the top-1000; their written run files truncate to 100


def rank_documents(doc_ids: Sequence[str], scores: Sequence[float],
                   top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Rank ``doc_ids`` by ``scores``, descending, breaking ties by corpus order.

    Mirrors `official_baseline.py`: a plain stable sort on score alone, so equal scores keep
    the order in which the documents appear in the corpus. Do **not** add a doc-id tiebreak.

    Returns:
        ``[(doc_id, score), ...]``, at most ``top_k`` long.
    """
    if len(doc_ids) != len(scores):
        raise ValueError(f"{len(doc_ids)} doc_ids but {len(scores)} scores")
    ordered = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ordered[:top_k]


def run_from_scores(scores_by_topic: dict[str, dict[str, float]],
                    doc_order: Sequence[str] | None = None,
                    top_k: int = TOP_K) -> dict[str, list[tuple[str, float]]]:
    """Turn ``{topic: {doc_id: score}}`` into ranked lists.

    Args:
        scores_by_topic: sparse scores per topic.
        doc_order: corpus order, used for the tie-break. When omitted, dict insertion order
            is used — which is only correct if the caller built the dicts in corpus order.
        top_k: cut per topic.
    """
    position = {d: i for i, d in enumerate(doc_order)} if doc_order is not None else None
    out = {}
    for topic_id, scores in scores_by_topic.items():
        items = list(scores.items())
        if position is not None:
            items.sort(key=lambda kv: position.get(kv[0], len(position)))
        ranked = sorted(items, key=lambda kv: kv[1], reverse=True)[:top_k]
        out[topic_id] = ranked
    return out


def write_run(path: Path, ranked: dict[str, list[tuple[str, float]]],
              tag: str = DEFAULT_TAG, top_k: int | None = None) -> Path:
    """Write a TREC run file. Returns the path, and says where it went (§11).

    Ranks are 1-based and unique within a topic, which the organizers' `format_checker.py`
    enforces. Topics are written in the order given; ties inside a topic must already be
    resolved by the caller (use ``rank_documents``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for topic_id, docs in ranked.items():
            rows = docs[:top_k] if top_k is not None else docs
            for rank, (doc_id, score) in enumerate(rows, start=1):
                handle.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {tag}\n")
    return path


def read_run(path: Path) -> dict[str, list[str]]:
    """Read a TREC run into ``{topic: [doc_id ordered by rank]}``.

    Sorts by ``(rank, -score)`` exactly as the organizers' `scorer.py::load_run` does, so a
    file whose lines are out of order still reads back correctly.
    """
    rows: dict[str, list[tuple[int, float, str]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            topic_id, _, doc_id, rank, score, _tag = line.split()
            rows.setdefault(topic_id, []).append((int(rank), float(score), doc_id))
    return {
        topic_id: [d for _, _, d in sorted(items, key=lambda r: (r[0], -r[1]))]
        for topic_id, items in rows.items()
    }


def read_run_with_scores(path: Path) -> dict[str, dict[str, float]]:
    """Read a TREC run into ``{topic: {doc_id: score}}``, for fusion and rescoring."""
    out: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            topic_id, _, doc_id, _rank, score, _tag = line.split()
            out.setdefault(topic_id, {})[doc_id] = float(score)
    return out


def iter_run_lines(ranked: dict[str, list[tuple[str, float]]],
                   tag: str = DEFAULT_TAG) -> Iterable[str]:
    """Yield formatted run lines without writing a file (useful in tests)."""
    for topic_id, docs in ranked.items():
        for rank, (doc_id, score) in enumerate(docs, start=1):
            yield f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {tag}"
