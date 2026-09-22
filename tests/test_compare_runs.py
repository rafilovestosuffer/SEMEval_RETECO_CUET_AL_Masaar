"""Run comparison: the union-recall and marginal-gold numbers decide Phase 5 spending."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from compare_runs import (marginal_gold, overlap_at_k,  # noqa: E402
                          recall_at_k, union_recall_at_k)


def test_overlap_is_jaccard_of_the_top_k() -> None:
    a = ["d1", "d2", "d3", "d4"]
    b = ["d3", "d4", "d5", "d6"]
    assert overlap_at_k(a, b, 4) == pytest.approx(2 / 6)
    assert overlap_at_k(a, a, 4) == 1.0
    assert overlap_at_k(a, ["z1", "z2"], 4) == 0.0


def test_overlap_respects_the_cut() -> None:
    """Documents below k must not count — the cut is the whole point."""
    a = ["d1", "d2", "d3"]
    b = ["d3", "d1", "d2"]
    assert overlap_at_k(a, b, 1) == 0.0      # d1 vs d3
    assert overlap_at_k(a, b, 3) == 1.0


def test_recall_counts_only_gold_within_k() -> None:
    gold = {"g1", "g2", "g3", "g4"}
    ranked = ["g1", "x", "g2", "y", "g3"]
    assert recall_at_k(ranked, gold, 3) == pytest.approx(0.5)     # g1, g2
    assert recall_at_k(ranked, gold, 5) == pytest.approx(0.75)    # + g3
    assert recall_at_k(ranked, set(), 5) == 0.0


def test_union_recall_is_at_least_either_arm() -> None:
    """The union is the ceiling any pooling strategy can reach."""
    gold = {"g1", "g2"}
    a, b = ["g1", "x"], ["g2", "y"]
    assert recall_at_k(a, gold, 2) == pytest.approx(0.5)
    assert recall_at_k(b, gold, 2) == pytest.approx(0.5)
    assert union_recall_at_k(a, b, gold, 2) == pytest.approx(1.0)


def test_union_recall_adds_nothing_when_one_arm_subsumes_the_other() -> None:
    """The negative case that tells us pooling is not worth it."""
    gold = {"g1", "g2"}
    strong, weak = ["g1", "g2", "x"], ["g1", "z", "w"]
    assert union_recall_at_k(strong, weak, gold, 3) == recall_at_k(strong, gold, 3)


def test_marginal_gold_counts_relevant_disagreement_only() -> None:
    """Disagreement on non-relevant documents must not count — that is just noise."""
    gold = {"g1", "g2"}
    a = ["g1", "noise_a"]
    b = ["g2", "noise_b"]
    assert marginal_gold(a, b, gold, 2) == 1      # g1
    assert marginal_gold(b, a, gold, 2) == 1      # g2

    # total disagreement, but none of it relevant
    a2, b2 = ["n1", "n2"], ["n3", "n4"]
    assert overlap_at_k(a2, b2, 2) == 0.0
    assert marginal_gold(a2, b2, gold, 2) == 0
    assert marginal_gold(b2, a2, gold, 2) == 0


def test_marginal_gold_is_zero_when_lists_agree() -> None:
    gold = {"g1"}
    assert marginal_gold(["g1", "x"], ["g1", "y"], gold, 2) == 0


def test_empty_run_is_handled() -> None:
    gold = {"g1"}
    assert recall_at_k([], gold, 10) == 0.0
    assert union_recall_at_k([], ["g1"], gold, 10) == pytest.approx(1.0)
    assert marginal_gold([], ["g1"], gold, 10) == 0
    assert overlap_at_k([], [], 10) == 0.0
