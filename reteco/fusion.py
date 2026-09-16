"""Rank and score fusion.

Two distinct uses in this project, and they are not the same problem:

1. **Hybrid first stage (Phase 4)** — combine a dense ranking with BM25 over the *same* topic.
   DIVER's recipe is `0.5 * dense + 0.5 * BM25` on min-max normalised scores, worth about +3
   nDCG@10 on BRIGHT (`notes/lit/diver.md`). RRF is the score-free alternative.

2. **Step fusion (Phase 5, hypothesis H2)** — combine the rankings of a query's *several 1b
   steps*, plus its whole-query 1a ranking, into one 1a ranking. Nobody has published this;
   it is the intended contribution (`notes/lit/SUMMARY.md`).

Both reduce to "merge several ranked lists", so they share the primitives below.

**Do not fuse Step-Only rankings.** Two papers independently show decomposition used *instead
of* the full query hurts: TEMPO's Step-Only scores 14.6 vs Query+Step 26.4, and ReasonIR found
LangChain decomposition dropping GRIT-7B from 20.4 to 17.3. `reteco.data.topics_1b` already
builds the Query+Step form; fuse those.

RRF reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods", SIGIR '09.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

__all__ = [
    "RRF_K",
    "minmax_normalise",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "max_score_fusion",
    "sum_score_fusion",
    "interpolate",
    "to_ranking",
]

# Cormack et al. 2009 use k = 60; it damps the influence of the very top ranks.
RRF_K = 60

Ranking = Sequence[str]                 # doc ids, best first
ScoreMap = Mapping[str, float]          # doc_id -> score


def minmax_normalise(scores: ScoreMap) -> dict[str, float]:
    """Scale scores to [0, 1]. A flat input maps to all-1.0, not a division by zero.

    Required before interpolating dense and BM25 scores, whose ranges are unrelated
    (BM25 is unbounded, cosine similarity is roughly [-1, 1]).
    """
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return {doc_id: 1.0 for doc_id in scores}
    span = high - low
    return {doc_id: (score - low) / span for doc_id, score in scores.items()}


def reciprocal_rank_fusion(rankings: Sequence[Ranking],
                           weights: Sequence[float] | None = None,
                           k: int = RRF_K) -> dict[str, float]:
    """Fuse ranked lists by ``sum_i w_i / (k + rank_i)``, rank 1-based.

    Score-free, so it needs no normalisation and is immune to incomparable score scales —
    the reason it is the default for combining retrievers that disagree about magnitude.
    A document missing from a list simply contributes nothing for that list.

    Args:
        rankings: ranked doc-id lists, best first.
        weights: per-list weights; defaults to 1.0 each.
        k: RRF damping constant (60 in the original paper).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")

    fused: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += weight / (k + rank)
    return dict(fused)


def weighted_score_fusion(score_maps: Sequence[ScoreMap],
                          weights: Sequence[float] | None = None,
                          normalise: bool = True) -> dict[str, float]:
    """Weighted sum of scores, min-max normalised per list by default.

    With two lists and equal weights this is DIVER's hybrid:
    ``0.5 * dense + 0.5 * BM25`` on normalised scores.

    A document absent from one list contributes 0 for it — after normalisation 0 is the
    bottom of that list's range, which is the intended "this retriever did not rank it".
    """
    if weights is None:
        weights = [1.0] * len(score_maps)
    if len(weights) != len(score_maps):
        raise ValueError(f"{len(score_maps)} score maps but {len(weights)} weights")

    prepared = [minmax_normalise(s) if normalise else dict(s) for s in score_maps]
    fused: dict[str, float] = defaultdict(float)
    for scores, weight in zip(prepared, weights):
        for doc_id, score in scores.items():
            fused[doc_id] += weight * score
    return dict(fused)


def interpolate(a: ScoreMap, b: ScoreMap, alpha: float = 0.5,
                normalise: bool = True) -> dict[str, float]:
    """``alpha * a + (1 - alpha) * b``. DIVER's hybrid is ``alpha = 0.5``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return weighted_score_fusion([a, b], weights=[alpha, 1.0 - alpha], normalise=normalise)


def max_score_fusion(score_maps: Sequence[ScoreMap],
                     normalise: bool = True) -> dict[str, float]:
    """Per-document maximum across lists.

    The natural choice for step fusion: a document is relevant to the query if it strongly
    answers *any* required step, and averaging would punish it for the steps it does not
    address. Also how DIVER scores a document that was split into several chunks.
    """
    prepared = [minmax_normalise(s) if normalise else dict(s) for s in score_maps]
    fused: dict[str, float] = {}
    for scores in prepared:
        for doc_id, score in scores.items():
            if doc_id not in fused or score > fused[doc_id]:
                fused[doc_id] = score
    return fused


def sum_score_fusion(score_maps: Sequence[ScoreMap],
                     normalise: bool = True) -> dict[str, float]:
    """Per-document sum across lists — rewards documents covering several steps.

    The counterpart to ``max_score_fusion``: where max asks "does this document answer any
    step well?", sum asks "does it answer many steps?". Which is right for 1a is exactly
    what H2 has to measure, since 1a gold is the union over the query's evidence.
    """
    return weighted_score_fusion(score_maps, weights=None, normalise=normalise)


def to_ranking(fused: ScoreMap, top_k: int | None = None,
               doc_order: Sequence[str] | None = None) -> list[tuple[str, float]]:
    """Sort fused scores into a ranking, optionally breaking ties by corpus order.

    Pass ``doc_order`` to match the organizers' tie-break (see `reteco.runs`); without it,
    ties fall back to dict insertion order.
    """
    items = list(fused.items())
    if doc_order is not None:
        position = {d: i for i, d in enumerate(doc_order)}
        items.sort(key=lambda kv: position.get(kv[0], len(position)))
    ranked = sorted(items, key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked
