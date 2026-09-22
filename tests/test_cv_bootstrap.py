"""Cross-validation folds and bootstrap intervals.

Small iteration counts keep the suite fast; the statistical properties being asserted are
structural (determinism, stratification, pairing), not precision of the interval.
"""

from __future__ import annotations

import pytest

import bootstrap
import cv


def make_scores(sizes: dict[str, int], value: float = 0.5) -> dict[str, dict[str, float]]:
    return {d: {f"{d}_t{i}": value for i in range(n)} for d, n in sizes.items()}


# ------------------------------------------------------------------------- folds --
def test_folds_are_stratified_by_domain() -> None:
    """Every fold must see every domain, or it cannot compute the full macro."""
    topics = {d: [f"{d}_{i}" for i in range(20)] for d in ("a", "b", "c")}
    folds = cv.make_folds(topics, n_folds=5)
    assert len(folds) == 5
    for fold in folds:
        assert set(fold) == {"a", "b", "c"}
        assert all(len(v) == 4 for v in fold.values())


def test_folds_partition_without_overlap_or_loss() -> None:
    topics = {"a": [f"a{i}" for i in range(17)]}  # deliberately not divisible by 5
    folds = cv.make_folds(topics, n_folds=5)
    seen = [t for fold in folds for t in fold.get("a", [])]
    assert sorted(seen) == sorted(topics["a"])
    assert len(seen) == len(set(seen))


def test_a_tiny_domain_still_reaches_as_many_folds_as_it_has_topics() -> None:
    """IOTA has ~10 topics; with 5 folds each gets 2. With 3 topics, only 3 folds get one."""
    folds = cv.make_folds({"iota": ["a", "b", "c"]}, n_folds=5)
    present = [f for f in folds if f.get("iota")]
    assert len(present) == 3


def test_folds_are_deterministic_and_order_independent() -> None:
    first = cv.make_folds({"a": ["x", "y", "z", "w"]}, n_folds=2, seed=7)
    second = cv.make_folds({"a": ["w", "z", "y", "x"]}, n_folds=2, seed=7)
    assert first == second, "ids are sorted before shuffling, so input order must not matter"


def test_different_seeds_give_different_partitions() -> None:
    a = cv.make_folds({"d": [f"t{i}" for i in range(20)]}, n_folds=4, seed=1)
    b = cv.make_folds({"d": [f"t{i}" for i in range(20)]}, n_folds=4, seed=2)
    assert a != b


def test_too_few_folds_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        cv.make_folds({"a": ["x"]}, n_folds=1)


def test_fold_scores_restricts_to_the_fold() -> None:
    scores = make_scores({"a": 4})
    fold = {"a": ["a_t0", "a_t1"]}
    assert set(cv.fold_scores(scores, fold)["a"]) == {"a_t0", "a_t1"}


# --------------------------------------------------------------------- bootstrap --
def test_bootstrap_on_constant_scores_has_zero_width() -> None:
    result = bootstrap.bootstrap_macro(make_scores({"a": 10, "b": 10}, 0.4), iters=200)
    assert result["macro"] == pytest.approx(0.4)
    assert result["ci_low"] == pytest.approx(0.4)
    assert result["ci_high"] == pytest.approx(0.4)


def test_domain_mode_gives_every_domain_equal_weight() -> None:
    """mode='domain': a 1000-topic domain must not outweigh a 10-topic one.

    This reproduces the organizers' baseline table (the Phase 1/2 gates), and was the
    default until 22 Sept 2026, when `evaluation.html` was found to define the leaderboard
    as a macro over queries instead. It is now opt-in.
    """
    scores = {
        "big": {f"b{i}": 1.0 for i in range(1000)},
        "small": {f"s{i}": 0.0 for i in range(10)},
    }
    result = bootstrap.bootstrap_macro(scores, iters=200, mode="domain")
    assert result["macro"] == pytest.approx(0.5)


def test_query_mode_is_the_default_and_weights_by_topic_count() -> None:
    """mode='query' is the leaderboard metric (CLAUDE.md §4) and must be the default."""
    scores = {
        "big": {f"b{i}": 1.0 for i in range(1000)},
        "small": {f"s{i}": 0.0 for i in range(10)},
    }
    assert bootstrap.DEFAULT_MACRO_MODE == "query"
    result = bootstrap.bootstrap_macro(scores, iters=200)
    assert result["macro"] == pytest.approx(1000 / 1010)
    assert result["mode"] == "query"


def test_an_unknown_macro_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        bootstrap._macro({"a": [1.0]}, mode="per-domain")


def test_cross_validate_reports_both_aggregations() -> None:
    """§4 requires reporting both; a gain can change sign between them."""
    scores = {
        "big": {f"b{i}": 1.0 for i in range(100)},
        "small": {f"s{i}": 0.0 for i in range(10)},
    }
    result = cv.cross_validate(scores, n_folds=2, iters=100)
    assert result["mode"] == "query"
    assert result["other_mode"] == "domain"
    assert result["overall_macro"] == pytest.approx(100 / 110)
    assert result["other_macro"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    scores = {"a": {f"t{i}": i / 10 for i in range(10)}}
    first = bootstrap.bootstrap_macro(scores, iters=300, seed=42)
    second = bootstrap.bootstrap_macro(scores, iters=300, seed=42)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]


def test_small_domains_get_wider_intervals() -> None:
    """§6: tiny domains are high-variance and the CI must show it, not hide it."""
    spread = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    scores = {
        "tiny": {f"t{i}": v for i, v in enumerate(spread[:3])},
        "large": {f"l{i}": spread[i % len(spread)] for i in range(300)},
    }
    result = bootstrap.bootstrap_macro(scores, iters=500)
    per_domain = result["per_domain"]
    tiny_width = per_domain["tiny"]["ci_high"] - per_domain["tiny"]["ci_low"]
    large_width = per_domain["large"]["ci_high"] - per_domain["large"]["ci_low"]
    assert tiny_width > large_width


def test_bootstrap_on_empty_input_does_not_crash() -> None:
    assert bootstrap.bootstrap_macro({})["macro"] == 0.0


def test_percentile_ci_brackets_the_median() -> None:
    low, high = bootstrap.percentile_ci([float(i) for i in range(100)])
    assert low < 50 < high


# ------------------------------------------------------------- paired comparison --
def test_paired_detects_a_uniform_improvement() -> None:
    better = make_scores({"a": 30, "b": 30}, 0.6)
    worse = make_scores({"a": 30, "b": 30}, 0.4)
    result = bootstrap.paired_bootstrap(better, worse, iters=300)
    assert result["delta"] == pytest.approx(0.2)
    assert result["significant"] is True


def test_paired_reports_no_difference_for_identical_systems() -> None:
    scores = make_scores({"a": 20}, 0.5)
    result = bootstrap.paired_bootstrap(scores, scores, iters=300)
    assert result["delta"] == pytest.approx(0.0)
    assert result["significant"] is False


def test_paired_uses_only_topics_both_systems_scored() -> None:
    """A system missing topics the other has would otherwise bias the difference."""
    a = {"d": {"t1": 1.0, "t2": 1.0}}
    b = {"d": {"t1": 0.0}}
    result = bootstrap.paired_bootstrap(a, b, iters=100)
    assert result["num_topics"] == 1


def test_paired_with_no_shared_topics_is_not_significant() -> None:
    result = bootstrap.paired_bootstrap({"d": {"x": 1.0}}, {"d": {"y": 0.0}}, iters=100)
    assert result["num_topics"] == 0
    assert result["significant"] is False


def test_noisy_small_difference_is_not_called_significant() -> None:
    """The guard §5 asks for: a gain inside the noise must not be reported as a gain."""
    a = {"d": {f"t{i}": float(i % 2) for i in range(20)}}
    b = {"d": {f"t{i}": float((i + 1) % 2) for i in range(20)}}
    result = bootstrap.paired_bootstrap(a, b, iters=500)
    assert result["delta"] == pytest.approx(0.0)
    assert result["significant"] is False


# ------------------------------------------------------------------ cross_validate --
def test_cross_validate_reports_folds_ci_and_domain_spread() -> None:
    result = cv.cross_validate(make_scores({"a": 20, "b": 20}, 0.5), n_folds=5, iters=200)
    assert len(result["fold_macros"]) == 5
    assert result["overall_macro"] == pytest.approx(0.5)
    assert result["fold_macro_stdev"] == pytest.approx(0.0)
    assert set(result["domain_spread"]) == {"a", "b"}
    assert result["domain_spread"]["a"]["folds_present"] == 5


def test_cross_validate_surfaces_an_unstable_domain() -> None:
    """A stable macro can hide a domain swinging wildly — the sd column must expose it."""
    scores = {
        "steady": {f"s{i}": 0.5 for i in range(40)},
        "erratic": {f"e{i}": float(i % 2) for i in range(10)},
    }
    result = cv.cross_validate(scores, n_folds=5, iters=100)
    assert result["domain_spread"]["steady"]["stdev"] == pytest.approx(0.0)
    assert result["domain_spread"]["erratic"]["stdev"] > 0.0


def test_render_includes_the_per_domain_table() -> None:
    text = cv.render(cv.cross_validate(make_scores({"iota": 10}, 0.3), n_folds=5, iters=100))
    assert "iota" in text and "CI" in text and "sd" in text


def test_step_level_ids_are_rejected() -> None:
    """Folding 1b step ids splits a query across folds — measured at 74.8% on real data."""
    steps = {"d": {"q1_step1": 0.1, "q1_step2": 0.2, "q2_step1": 0.3, "q2_step2": 0.4}}
    with pytest.raises(ValueError, match="STEP ids"):
        cv.cross_validate(steps, n_folds=2, iters=50)


def test_query_level_ids_are_accepted() -> None:
    """The official 1b aggregation gives one value per query, which folds correctly."""
    queries = {"d": {f"q{i}": 0.1 * i for i in range(10)}}
    result = cv.cross_validate(queries, n_folds=2, iters=50)
    assert result["n_folds"] == 2


def test_a_single_step_per_query_is_not_flagged() -> None:
    """One step per query is unambiguous — no parent is ever split, so allow it."""
    single = {"d": {f"q{i}_step1": 0.1 for i in range(6)}}
    assert cv.cross_validate(single, n_folds=2, iters=50)["n_folds"] == 2
