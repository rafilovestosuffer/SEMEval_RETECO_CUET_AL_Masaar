"""Fusion primitives — Phase 5's core, and the module H2 rests on."""

from __future__ import annotations

import pytest

from reteco import fusion


# ------------------------------------------------------------------- normalisation --
def test_minmax_maps_to_unit_interval() -> None:
    out = fusion.minmax_normalise({"a": 10.0, "b": 20.0, "c": 15.0})
    assert out == pytest.approx({"a": 0.0, "b": 1.0, "c": 0.5})


def test_minmax_on_flat_input_does_not_divide_by_zero() -> None:
    assert fusion.minmax_normalise({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}


def test_minmax_on_empty_is_empty() -> None:
    assert fusion.minmax_normalise({}) == {}


def test_minmax_handles_negative_scores() -> None:
    """Cosine similarity can be negative; BM25 cannot. Both must normalise."""
    out = fusion.minmax_normalise({"a": -1.0, "b": 0.0, "c": 1.0})
    assert out == pytest.approx({"a": 0.0, "b": 0.5, "c": 1.0})


# ---------------------------------------------------------------------------- RRF --
def test_rrf_matches_the_published_formula() -> None:
    """Cormack et al. 2009: sum of 1/(k + rank), rank 1-based."""
    fused = fusion.reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    expected = 1 / 61 + 1 / 62  # both docs hold rank 1 in one list and 2 in the other
    assert fused["a"] == pytest.approx(expected)
    assert fused["b"] == pytest.approx(expected)


def test_rrf_rewards_consistent_high_rank() -> None:
    fused = fusion.reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    ranking = [d for d, _ in fusion.to_ranking(fused)]
    assert ranking[0] == "a"


def test_rrf_document_missing_from_a_list_contributes_nothing() -> None:
    fused = fusion.reciprocal_rank_fusion([["a"], ["b"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_rrf_weights_are_applied() -> None:
    fused = fusion.reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0], k=60)
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 61)
    assert fusion.to_ranking(fused)[0][0] == "a"


def test_rrf_smaller_k_sharpens_top_rank_influence() -> None:
    """k damps the top ranks; a smaller k widens the rank-1 vs rank-2 gap."""
    ranking = ["a", "b"]
    wide = fusion.reciprocal_rank_fusion([ranking], k=1)
    narrow = fusion.reciprocal_rank_fusion([ranking], k=1000)
    assert (wide["a"] - wide["b"]) > (narrow["a"] - narrow["b"])


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        fusion.reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


# ----------------------------------------------------------------- score fusion --
def test_interpolate_is_the_diver_hybrid() -> None:
    """DIVER: 0.5 * dense + 0.5 * BM25 on min-max normalised scores."""
    dense = {"a": 1.0, "b": 0.0}
    sparse = {"a": 0.0, "b": 1.0}
    fused = fusion.interpolate(dense, sparse, alpha=0.5)
    assert fused["a"] == pytest.approx(0.5)
    assert fused["b"] == pytest.approx(0.5)


def test_interpolate_alpha_one_returns_the_first_list() -> None:
    dense = {"a": 10.0, "b": 0.0}
    fused = fusion.interpolate(dense, {"a": 0.0, "b": 99.0}, alpha=1.0)
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(0.0)


def test_interpolate_rejects_alpha_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fusion.interpolate({"a": 1.0}, {"a": 1.0}, alpha=1.5)


def test_normalisation_is_what_makes_incomparable_scales_fusable() -> None:
    """BM25 is unbounded, cosine is ~[-1,1]. Unnormalised, BM25 would dominate entirely."""
    bm25 = {"a": 40.0, "b": 10.0}
    dense = {"a": 0.1, "b": 0.9}
    raw = fusion.weighted_score_fusion([bm25, dense], normalise=False)
    normed = fusion.weighted_score_fusion([bm25, dense], normalise=True)
    assert raw["a"] > raw["b"]          # BM25's magnitude swamps the dense signal
    assert normed["a"] == pytest.approx(normed["b"])  # after scaling they tie


# ------------------------------------------------------------ max vs sum (H2) --
def test_max_takes_the_best_single_step() -> None:
    """A doc answering one step superbly should not be punished for the others."""
    step1 = {"a": 1.0, "b": 0.0}
    step2 = {"a": 0.0, "b": 1.0}
    fused = fusion.max_score_fusion([step1, step2])
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(1.0)


def test_sum_rewards_covering_several_steps() -> None:
    """The distinction max vs sum exists to measure: H2 asks which suits 1a gold."""
    step1 = {"a": 1.0, "b": 0.5}
    step2 = {"a": 1.0, "b": 0.5}
    summed = fusion.sum_score_fusion([step1, step2], normalise=False)
    maxed = fusion.max_score_fusion([step1, step2], normalise=False)
    assert summed["a"] == pytest.approx(2.0)
    assert maxed["a"] == pytest.approx(1.0)
    # sum separates a broad-coverage doc from a narrow one more than max does
    assert (summed["a"] - summed["b"]) > (maxed["a"] - maxed["b"])


def test_max_and_sum_disagree_on_ordering() -> None:
    """The case that makes H2 an empirical question rather than a design choice."""
    broad = [{"broad": 0.6, "narrow": 1.0}, {"broad": 0.6, "narrow": 0.0}]
    assert fusion.to_ranking(fusion.max_score_fusion(broad, normalise=False))[0][0] == "narrow"
    assert fusion.to_ranking(fusion.sum_score_fusion(broad, normalise=False))[0][0] == "broad"


# ------------------------------------------------------------------- to_ranking --
def test_to_ranking_sorts_descending_and_cuts() -> None:
    ranked = fusion.to_ranking({"a": 0.1, "b": 0.9, "c": 0.5}, top_k=2)
    assert [d for d, _ in ranked] == ["b", "c"]


def test_to_ranking_breaks_ties_by_corpus_order() -> None:
    """Matches the organizers' stable sort — never by doc id (see reteco.runs)."""
    tied = {"z": 1.0, "a": 1.0}
    ranked = fusion.to_ranking(tied, doc_order=["z", "a"])
    assert [d for d, _ in ranked] == ["z", "a"], "corpus order must win, not alphabetical"


def test_to_ranking_places_unknown_docs_last_under_corpus_order() -> None:
    ranked = fusion.to_ranking({"ghost": 1.0, "real": 1.0}, doc_order=["real"])
    assert [d for d, _ in ranked] == ["real", "ghost"]
