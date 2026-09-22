"""Dense retrieval: the deduplicated index must behave exactly like a full one."""

from __future__ import annotations

import json

import numpy as np
import pytest

from reteco.dense import QUERY_PROMPT, load_domain, search


def write_domain(tmp_path, vectors: np.ndarray, doc_ids: list[str], rows: list[int],
                 shards: int = 1):
    """Write a Phase 4b-shaped domain directory."""
    directory = tmp_path / "dom"
    directory.mkdir(exist_ok=True)
    for i, part in enumerate(np.array_split(vectors, shards)):
        np.save(directory / f"emb_{i:03d}.npy", part.astype(np.float16))
    (directory / "doc_index.json").write_text(
        json.dumps({"doc_ids": doc_ids, "rows": rows, "n_unique": len(vectors)}),
        encoding="utf-8")
    return directory


def unit(rows: list[list[float]]) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_shards_concatenate_in_order(tmp_path) -> None:
    vecs = unit([[1, 0], [0, 1], [1, 1], [1, -1]])
    d = write_domain(tmp_path, vecs, ["a", "b", "c", "d"], [0, 1, 2, 3], shards=3)
    index = load_domain(d)
    assert index.n_unique == 4
    np.testing.assert_allclose(index.vectors, vecs, atol=1e-3)


def test_duplicates_share_a_row_and_score_identically(tmp_path) -> None:
    """The saving must be invisible: a duplicate is ranked, not dropped."""
    vecs = unit([[1, 0], [0, 1]])
    d = write_domain(tmp_path, vecs, ["a", "b", "c"], [0, 1, 0])  # a and c identical
    index = load_domain(d)
    ranked = search(index, unit([[1, 0]]), ["q"])
    scores = dict(ranked["q"])
    assert set(scores) == {"a", "b", "c"}
    assert scores["a"] == pytest.approx(scores["c"])
    assert scores["a"] > scores["b"]


def test_ties_break_by_corpus_order(tmp_path) -> None:
    """Duplicates are exact ties by construction — far more common than with BM25."""
    vecs = unit([[1, 0]])
    d = write_domain(tmp_path, vecs, ["z_first", "m_second", "a_third"], [0, 0, 0])
    ranked = search(load_domain(d), unit([[1, 0]]), ["q"])
    assert [doc for doc, _ in ranked["q"]] == ["z_first", "m_second", "a_third"], (
        "equal scores must keep corpus order, not sort by doc id"
    )


def test_ranking_is_by_descending_similarity(tmp_path) -> None:
    vecs = unit([[1, 0], [0.8, 0.6], [0, 1]])
    d = write_domain(tmp_path, vecs, ["near", "mid", "far"], [0, 1, 2])
    ranked = search(load_domain(d), unit([[1, 0]]), ["q"])
    assert [doc for doc, _ in ranked["q"]] == ["near", "mid", "far"]
    assert ranked["q"][0][1] == pytest.approx(1.0, abs=1e-3)


def test_top_k_truncates(tmp_path) -> None:
    vecs = unit([[1, 0], [0.9, 0.1], [0.8, 0.2]])
    d = write_domain(tmp_path, vecs, ["a", "b", "c"], [0, 1, 2])
    assert len(search(load_domain(d), unit([[1, 0]]), ["q"], top_k=2)["q"]) == 2


def test_blocking_does_not_change_the_ranking(tmp_path) -> None:
    """Block size is a memory knob and must not reorder results.

    Scores are *not* bit-identical across block sizes: BLAS picks different accumulation
    orders for different matrix shapes, which moves the last couple of float32 bits
    (~1e-7 observed). The ranking is what must be stable, so that is what is asserted.
    See `reteco.dense.search` for the determinism caveat this implies.
    """
    rng = np.random.default_rng(0)
    vecs = unit(rng.normal(size=(40, 8)).tolist())
    d = write_domain(tmp_path, vecs, [f"d{i}" for i in range(40)], list(range(40)))
    index = load_domain(d)
    qs = unit(rng.normal(size=(7, 8)).tolist())
    qids = [f"q{i}" for i in range(7)]

    small, large = search(index, qs, qids, block=1), search(index, qs, qids, block=64)
    assert set(small) == set(large)
    for qid in qids:
        assert [d for d, _ in small[qid]] == [d for d, _ in large[qid]]
        np.testing.assert_allclose([s for _, s in small[qid]],
                                   [s for _, s in large[qid]], atol=1e-5)


def test_queries_are_normalised_before_scoring(tmp_path) -> None:
    """An unnormalised query must rank the same as its normalised twin."""
    vecs = unit([[1, 0], [0, 1]])
    d = write_domain(tmp_path, vecs, ["a", "b"], [0, 1])
    index = load_domain(d)
    scaled = search(index, np.array([[5.0, 0.0]], dtype=np.float32), ["q"])
    once = search(index, unit([[1, 0]]), ["q"])
    assert [k for k, _ in scaled["q"]] == [k for k, _ in once["q"]]
    assert scaled["q"][0][1] == pytest.approx(once["q"][0][1], abs=1e-4)


def test_shard_count_mismatch_is_caught(tmp_path) -> None:
    vecs = unit([[1, 0], [0, 1]])
    d = write_domain(tmp_path, vecs, ["a", "b"], [0, 1])
    meta = json.loads((d / "doc_index.json").read_text())
    meta["n_unique"] = 99
    (d / "doc_index.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="doc_index says 99"):
        load_domain(d)


def test_non_unit_vectors_are_caught(tmp_path) -> None:
    """A truncated or corrupt shard would otherwise distort every score silently."""
    d = write_domain(tmp_path, np.array([[3.0, 4.0]], dtype=np.float32), ["a"], [0])
    with pytest.raises(ValueError, match="not unit-norm"):
        load_domain(d)


def test_missing_index_and_shards_are_clear_errors(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="doc_index.json"):
        load_domain(empty)


def test_mismatched_query_inputs_are_rejected(tmp_path) -> None:
    vecs = unit([[1, 0]])
    index = load_domain(write_domain(tmp_path, vecs, ["a"], [0]))
    with pytest.raises(ValueError, match="query ids"):
        search(index, unit([[1, 0], [0, 1]]), ["only-one"])
    with pytest.raises(ValueError, match="query dim"):
        search(index, unit([[1, 0, 0]]), ["q"])


def test_query_prompt_matches_the_model_card() -> None:
    """Documents are embedded with no prefix; only queries carry this one."""
    assert QUERY_PROMPT.startswith("Instruct:")
    assert QUERY_PROMPT.endswith("Query:")
