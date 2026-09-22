"""Step fusion (Phase 5, H2). The parent weight is the knob the contribution rests on."""

from __future__ import annotations

import pytest

from reteco.stepfuse import (fuse_query_and_steps, fuse_run, group_steps,
                             normalise_scores, parent_of)


def test_parent_of_strips_the_step_suffix() -> None:
    assert parent_of("367_805_step1") == "367_805"
    assert parent_of("367_805_step12") == "367_805"
    assert parent_of("367_805") == "367_805", "a query id is its own parent"


def test_group_steps_collects_by_parent_deterministically() -> None:
    run = {"q2_step1": [], "q1_step2": [], "q1_step1": []}
    assert group_steps(run) == {"q1": ["q1_step1", "q1_step2"], "q2": ["q2_step1"]}


# ----------------------------------------------------------------- normalisation --
def test_theoretical_normalisation_keeps_a_fixed_lower_bound() -> None:
    """The point of 'theoretical': a weak list stays weak instead of being stretched."""
    strong = normalise_scores([("a", 10.0), ("b", 9.0)], "theoretical")
    weak = normalise_scores([("c", 0.02), ("d", 0.01)], "theoretical")
    assert strong["a"] == pytest.approx(1.0)
    assert weak["c"] == pytest.approx(1.0)          # top of its own list
    assert weak["d"] == pytest.approx(0.5)          # but the gap is preserved
    assert strong["b"] == pytest.approx(0.9)


def test_minmax_stretches_a_degenerate_list_and_theoretical_does_not() -> None:
    """This is the failure mode that makes empirical min-max dangerous here."""
    nearly_equal = [("c", 0.011), ("d", 0.010)]
    mm = normalise_scores(nearly_equal, "minmax")
    th = normalise_scores(nearly_equal, "theoretical")
    assert mm["c"] == 1.0 and mm["d"] == 0.0, "min-max spans the full range regardless"
    assert th["d"] == pytest.approx(0.909, abs=1e-3), "theoretical keeps them close"


def test_flat_list_does_not_divide_by_zero() -> None:
    assert normalise_scores([("a", 5.0), ("b", 5.0)], "minmax") == {"a": 1.0, "b": 1.0}
    assert normalise_scores([], "minmax") == {}


def test_unknown_normalisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown normalisation"):
        normalise_scores([("a", 1.0)], "zscore-ish")


# ------------------------------------------------------------------- the weight --
def test_large_parent_weight_recovers_whole_query_retrieval() -> None:
    """One endpoint of the sweep must reproduce the 1a baseline exactly."""
    query = [("a", 1.0), ("b", 0.5)]
    steps = [[("c", 1.0), ("d", 0.9)]]
    fused = fuse_query_and_steps(query, steps, parent_weight=1e6)
    assert max(fused, key=fused.get) == "a"
    assert fused["a"] > fused["c"], "steps must not outrank the query at high parent weight"


def test_zero_parent_weight_is_step_only() -> None:
    """The other endpoint is TEMPO's known-bad Step-Only condition — useful as a control."""
    query = [("a", 1.0)]
    steps = [[("c", 1.0)]]
    fused = fuse_query_and_steps(query, steps, parent_weight=0.0)
    assert "a" not in fused and fused["c"] == pytest.approx(1.0)


def test_parent_influence_is_scale_free_in_step_count() -> None:
    """parent_weight=n_steps should give the parent half the total weight, at any n."""
    query = [("p", 1.0)]
    for n in (1, 3, 5):
        steps = [[("s", 1.0)] for _ in range(n)]
        fused = fuse_query_and_steps(query, steps, parent_weight=float(n), operator="sum")
        assert fused["p"] == pytest.approx(0.5), f"n={n}"
        assert fused["s"] == pytest.approx(0.5), f"n={n}"


def test_negative_parent_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="parent_weight"):
        fuse_query_and_steps([("a", 1.0)], [], parent_weight=-1.0)


# --------------------------------------------------------------------- operators --
def test_sum_rewards_documents_covering_several_steps() -> None:
    """Complementary evidence: a document answering two steps should beat one answering one."""
    query = [("x", 0.0)]
    steps = [[("both", 1.0), ("one", 1.0)], [("both", 1.0), ("other", 1.0)]]
    fused = fuse_query_and_steps(query, steps, parent_weight=0.0, operator="sum")
    assert fused["both"] > fused["one"] and fused["both"] > fused["other"]


def test_max_does_not_reward_coverage() -> None:
    """The contrast that makes sum-vs-max an informative ablation, not a detail."""
    query = [("x", 0.0)]
    steps = [[("both", 1.0), ("one", 1.0)], [("both", 1.0), ("other", 1.0)]]
    fused = fuse_query_and_steps(query, steps, parent_weight=0.0, operator="max")
    assert fused["both"] == pytest.approx(fused["one"])


def test_rrf_ignores_score_magnitude() -> None:
    """Why RRF is risky here: a list that found nothing still contributes 1/(k+1)."""
    query = [("good", 100.0)]
    steps = [[("junk", 0.0000001)]]
    fused = fuse_query_and_steps(query, steps, parent_weight=1.0, operator="rrf")
    assert fused["good"] == pytest.approx(fused["junk"]), (
        "RRF cannot tell a confident list from an empty one — that is the hazard"
    )
    summed = fuse_query_and_steps(query, steps, parent_weight=1.0, operator="sum",
                                  normalise="none")
    assert summed["good"] > summed["junk"]


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown operator"):
        fuse_query_and_steps([("a", 1.0)], [], operator="borda")


# ------------------------------------------------------------------------- runs --
def test_fuse_run_leaves_stepless_queries_alone() -> None:
    """A query without a decomposition must not be penalised for lacking one."""
    query_run = {"q1": [("a", 1.0), ("b", 0.5)]}
    fused = fuse_run(query_run, step_run={}, parent_weight=1.0)
    assert [d for d, _ in fused["q1"]] == ["a", "b"]


def test_fuse_run_matches_steps_to_their_own_parent() -> None:
    query_run = {"q1": [("a", 1.0)], "q2": [("b", 1.0)]}
    step_run = {"q1_step1": [("from_q1", 1.0)], "q2_step1": [("from_q2", 1.0)]}
    fused = fuse_run(query_run, step_run, parent_weight=0.0)
    assert [d for d, _ in fused["q1"]] == ["from_q1"]
    assert [d for d, _ in fused["q2"]] == ["from_q2"]


def test_fuse_run_breaks_ties_by_corpus_order() -> None:
    """Fused scores tie often, since duplicates share an embedding row."""
    query_run = {"q1": [("z", 1.0), ("m", 1.0), ("a", 1.0)]}
    fused = fuse_run(query_run, {}, doc_order=["z", "m", "a"], parent_weight=1.0)
    assert [d for d, _ in fused["q1"]] == ["z", "m", "a"]


def test_fuse_run_truncates() -> None:
    query_run = {"q1": [(f"d{i}", 1.0 - i * 0.01) for i in range(50)]}
    assert len(fuse_run(query_run, {}, parent_weight=1.0, top_k=10)["q1"]) == 10
