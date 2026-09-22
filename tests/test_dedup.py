"""Deduplication is pure bookkeeping — it must never change which vector a document gets."""

from __future__ import annotations

import pytest

from reteco.dedup import DedupPlan, build_plan, content_key


def test_identical_text_shares_one_encode_slot() -> None:
    plan = build_plan(["a", "b", "c"], ["same", "other", "same"], sort_by_length=False)
    assert len(plan.unique_texts) == 2
    assert plan.doc_to_unique["a"] == plan.doc_to_unique["c"]
    assert plan.doc_to_unique["b"] != plan.doc_to_unique["a"]
    assert plan.duplicates == 1


def test_every_document_keeps_a_slot_including_duplicates() -> None:
    """A duplicate must resolve to a row, not be dropped — the run file needs all ids."""
    ids = [f"d{i}" for i in range(10)]
    plan = build_plan(ids, ["x"] * 10)
    assert set(plan.doc_to_unique) == set(ids)
    assert len(plan.unique_texts) == 1
    assert {plan.doc_to_unique[i] for i in ids} == {0}
    assert plan.saving == pytest.approx(0.9)


def test_sorting_does_not_break_the_mapping() -> None:
    """The permutation is where this could silently corrupt every ranking."""
    ids = ["a", "b", "c", "d"]
    texts = ["wwwwwwwwww", "x", "wwwwwwwwww", "yy"]
    plan = build_plan(ids, texts, sort_by_length=True)
    for doc_id, original in zip(ids, texts):
        assert plan.unique_texts[plan.doc_to_unique[doc_id]] == original, doc_id


def test_unique_texts_are_ordered_shortest_first() -> None:
    plan = build_plan(["a", "b", "c"], ["ccc", "a", "bb"])
    assert plan.unique_texts == ["a", "bb", "ccc"]
    assert plan.order_proxy == [1, 2, 3]


def test_whitespace_differences_are_not_collapsed() -> None:
    """Different bytes are different encoder inputs; normalising would be a modelling call."""
    plan = build_plan(["a", "b"], ["hello world", "hello  world"], sort_by_length=False)
    assert len(plan.unique_texts) == 2
    assert plan.duplicates == 0


def test_case_differences_are_not_collapsed() -> None:
    plan = build_plan(["a", "b"], ["Text", "text"], sort_by_length=False)
    assert len(plan.unique_texts) == 2


def test_mismatched_inputs_are_rejected_rather_than_misaligned() -> None:
    with pytest.raises(ValueError, match="2 doc_ids but 3 texts"):
        build_plan(["a", "b"], ["x", "y", "z"])


def test_empty_corpus() -> None:
    plan = build_plan([], [])
    assert len(plan) == 0 and plan.saving == 0.0


def test_content_key_is_stable_and_distinguishing() -> None:
    assert content_key("abc") == content_key("abc")
    assert content_key("abc") != content_key("abd")
    assert len(content_key("abc")) == 32  # blake2b digest_size=16 -> 32 hex chars


def test_plan_reconstructs_the_original_corpus_texts() -> None:
    """End-to-end invariant: scattering the unique rows back must rebuild every document."""
    ids = [f"d{i}" for i in range(50)]
    texts = [f"body {i % 7}" for i in range(50)]  # 7 distinct, 43 duplicates
    plan = build_plan(ids, texts)
    assert len(plan.unique_texts) == 7
    assert plan.duplicates == 43
    rebuilt = [plan.unique_texts[plan.doc_to_unique[i]] for i in ids]
    assert rebuilt == texts
