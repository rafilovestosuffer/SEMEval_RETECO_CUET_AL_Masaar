"""Fuse sub-track 1b step rankings into a sub-track 1a query ranking (Phase 5, H2).

The design follows what the literature actually measured, not what sounds elegant
(`reports/Temporal query and fusion design.md`):

- **Augment, never replace.** Every published positive result keeps the whole-query
  ranking and *adds* the sub-query rankings; every published negative uses sub-queries
  instead of the query. TEMPO's own Step-Only is 14.6 against Query+Step 26.4, and a
  stage-aware study found decomposition costing 1.6-4.3 nDCG@10 at first stage when it
  replaces the query. So the parent list is always a participant here, and
  ``parent_weight`` controls how loud it is rather than whether it is present.
- **Sum, not RRF.** ReDI's ablation on BRIGHT StackExchange: Sum 38.3 > RRF 36.8 > Max
  35.2 > Concat 30.7 (sparse arm). Two other decomposition papers independently land on
  additive aggregation. RRF's published wins come from fusing *substitutable* query
  variants, which is a different problem — see the note on semantics below.
- **RRF is actively risky in our regime.** It ignores score magnitude, so a step that
  retrieved nothing relevant still contributes a full ``1/(k+1)`` at its rank 1. With
  77.7% of our BM25 queries scoring exactly zero, that launders noise into confident
  ranks. It is implemented here for the ablation, not recommended.

**Why the parent weight is the interesting knob.** Views and reformulations are
*substitutable* — alternatives phrasing one need, so a reliability-weighted average is
natural. Decomposed steps are *complementary* — each covers different evidence, and the
union is the answer. No reviewed paper sweeps the parent-versus-children weight, which is
what makes it a measurable contribution rather than a reimplementation. The sweep also has
clean endpoints: ``parent_weight`` large recovers whole-query retrieval exactly, and
``parent_weight=0`` is the known-bad Step-Only condition, so the curve is interpretable
at both ends.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

__all__ = ["STEP_MARKER", "parent_of", "group_steps", "normalise_scores",
           "fuse_query_and_steps", "fuse_run"]

STEP_MARKER = "_step"

Ranked = Sequence[tuple[str, float]]
RunLike = Mapping[str, Ranked]


def parent_of(step_id: str) -> str:
    """``367_805_step1`` -> ``367_805``; an id without the marker is its own parent."""
    return step_id.rsplit(STEP_MARKER, 1)[0] if STEP_MARKER in step_id else step_id


def group_steps(step_run: RunLike) -> dict[str, list[str]]:
    """``{parent_query_id: [step_id, ...]}``, steps in sorted order for determinism."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for step_id in step_run:
        grouped[parent_of(step_id)].append(step_id)
    return {q: sorted(s) for q, s in grouped.items()}


def normalise_scores(ranked: Ranked, method: str = "minmax",
                     floor: float | None = None) -> dict[str, float]:
    """Map one ranked list onto a comparable scale.

    Args:
        method: ``"minmax"`` scales to [0, 1] using this list's own extremes;
            ``"theoretical"`` divides by the list maximum only, keeping 0 as a fixed lower
            bound; ``"rank"`` ignores scores entirely and uses ``1/(rank)``; ``"none"``
            passes scores through.
        floor: score assigned to documents absent from this list when it is combined with
            others. Defaults to 0.0.

    ``"theoretical"`` is the safer default when a list may be degenerate. Empirical
    min-max subtracts the list's own minimum, so a list where every score is tiny and
    nearly equal gets stretched to span [0, 1] and contributes as confidently as a list
    that genuinely found something. That is exactly the failure mode our weak arm has.
    """
    items = list(ranked)
    if not items:
        return {}
    scores = [s for _, s in items]

    if method == "none":
        return {d: float(s) for d, s in items}
    if method == "rank":
        return {d: 1.0 / (i + 1) for i, (d, _) in enumerate(items)}
    if method == "theoretical":
        top = max(scores)
        if top <= 0:
            return {d: 0.0 for d, _ in items}
        return {d: float(s) / top for d, s in items}
    if method == "minmax":
        low, high = min(scores), max(scores)
        if high == low:
            return {d: 1.0 for d, _ in items}
        return {d: (float(s) - low) / (high - low) for d, s in items}
    raise ValueError(f"unknown normalisation {method!r}")


def fuse_query_and_steps(query_ranked: Ranked, step_ranked: Sequence[Ranked],
                         parent_weight: float = 1.0, operator: str = "sum",
                         normalise: str = "theoretical",
                         missing: float = 0.0,
                         rrf_k: int = 60) -> dict[str, float]:
    """Combine one query's whole-query list with its step lists.

    Args:
        parent_weight: weight on the whole-query list; each step gets weight 1. The
            parent's influence is therefore ``parent_weight / (parent_weight + n_steps)``,
            so the knob is scale-free in the number of steps.
        operator: ``"sum"`` (recommended), ``"max"``, or ``"rrf"`` (ablation only).
        missing: score for a document absent from a given list. 0.0 under a normalisation
            with a fixed lower bound; a negative value penalises absence, which matters
            when a document appearing in one list and nowhere else should not outrank one
            that appears everywhere.

    Returns:
        ``{doc_id: fused_score}``. Ranking and truncation are the caller's job, so the
        organizers' corpus-order tie-break is applied once, centrally.
    """
    if parent_weight < 0:
        raise ValueError(f"parent_weight must be >= 0, got {parent_weight}")

    if operator == "rrf":
        fused: dict[str, float] = defaultdict(float)
        for weight, ranked in [(parent_weight, query_ranked)] + [(1.0, r) for r in step_ranked]:
            for rank, (doc, _) in enumerate(ranked, start=1):
                fused[doc] += weight / (rrf_k + rank)
        return dict(fused)

    lists = [(parent_weight, normalise_scores(query_ranked, normalise))]
    lists += [(1.0, normalise_scores(r, normalise)) for r in step_ranked]
    lists = [(w, s) for w, s in lists if s and w > 0]
    if not lists:
        return {}

    docs = {d for _, s in lists for d in s}
    if operator == "max":
        return {d: max(s.get(d, missing) for _, s in lists) for d in docs}
    if operator == "sum":
        total = sum(w for w, _ in lists)
        return {d: sum(w * s.get(d, missing) for w, s in lists) / total for d in docs}
    raise ValueError(f"unknown operator {operator!r}")


def fuse_run(query_run: RunLike, step_run: RunLike, doc_order: Sequence[str] | None = None,
             parent_weight: float = 1.0, operator: str = "sum",
             normalise: str = "theoretical", missing: float = 0.0,
             rrf_k: int = 60, top_k: int = 100) -> dict[str, list[tuple[str, float]]]:
    """Fuse a whole 1b run into a 1a run, one parent query at a time.

    Queries present in ``query_run`` but with no steps pass through fused with nothing,
    which leaves their ranking unchanged — the right behaviour, since a query without a
    decomposition should not be penalised for it.

    Args:
        doc_order: corpus order for the tie-break. Fused scores tie often (duplicate
            documents share an embedding row and therefore score identically), so without
            this the order of equal-scoring documents depends on dict iteration.
    """
    from reteco.runs import rank_documents

    grouped = group_steps(step_run)
    position = {d: i for i, d in enumerate(doc_order)} if doc_order is not None else None

    out: dict[str, list[tuple[str, float]]] = {}
    for qid, ranked in query_run.items():
        steps = [list(step_run[s]) for s in grouped.get(qid, [])]
        fused = fuse_query_and_steps(list(ranked), steps, parent_weight=parent_weight,
                                     operator=operator, normalise=normalise,
                                     missing=missing, rrf_k=rrf_k)
        items = list(fused.items())
        if position is not None:
            items.sort(key=lambda kv: position.get(kv[0], len(position)))
        docs = [d for d, _ in items]
        scores = [s for _, s in items]
        out[qid] = rank_documents(docs, scores, top_k=top_k)
    return out
