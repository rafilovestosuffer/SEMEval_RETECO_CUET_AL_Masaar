"""Schema loaders and TREC run handling, against the synthetic fixture.

The fixture (`tests/fixtures/make_fixture.py`) is written in the real release schema, so these
tests exercise the same field names `official_baseline.py` reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reteco import data, runs

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIXTURE_ROOT))


@pytest.fixture(scope="module")
def domain_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a synthetic IOTA-shaped domain once for the module."""
    from make_fixture import build

    out = tmp_path_factory.mktemp("reteco_data")
    return build(out, "iota", n_docs=60, n_queries=10, gold_per_query=4, steps_per_query=2)


# ------------------------------------------------------------------------- corpus --
def test_corpus_preserves_file_order(domain_path: Path) -> None:
    """Order is load-bearing: it is the organizers' tie-break."""
    corpus = data.load_corpus("iota", root=domain_path)
    lines = (domain_path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    import json
    assert corpus.doc_ids == [json.loads(ln)["id"] for ln in lines if ln.strip()]
    assert len(corpus) == 60


def test_corpus_position_map_matches_file_order(domain_path: Path) -> None:
    corpus = data.load_corpus("iota", root=domain_path)
    position = corpus.position()
    assert position[corpus.doc_ids[0]] == 0
    assert position[corpus.doc_ids[-1]] == len(corpus) - 1


# ------------------------------------------------------------------------- topics --
def test_topics_1a_use_the_raw_query_and_id(domain_path: Path) -> None:
    topics = data.topics_1a("iota", "train", domain_path)
    examples = data.load_examples("iota", "train", domain_path)
    assert [t.topic_id for t in topics] == [e["id"] for e in examples]
    assert [t.text for t in topics] == [e["query"] for e in examples]


def test_topics_1b_use_step_instruction_not_step(domain_path: Path) -> None:
    """official_baseline.py's docstring says 'step_text'; its code uses step_instruction."""
    records = data.load_steps("iota", "train", domain_path)
    topics = data.topics_1b("iota", "train", domain_path)
    first_step = records[0]["steps"][0]

    assert topics[0].topic_id == first_step["step_id"]
    assert first_step["step_instruction"] in topics[0].text
    assert first_step["step"] not in topics[0].text, "must use step_instruction, not step"


def test_1b_query_template_is_exact() -> None:
    assert data.build_1b_query("Q", "I") == "Q\n\nStep: I"


def test_topics_1b_records_its_parent_query(domain_path: Path) -> None:
    """Needed by Phase 5: step rankings fuse back into the parent query's 1a ranking."""
    topics = data.topics_1b("iota", "train", domain_path)
    records = data.load_steps("iota", "train", domain_path)
    assert topics[0].parent_id == records[0]["id"]
    assert len(topics) == sum(len(r["steps"]) for r in records)


# -------------------------------------------------------------------------- qrels --
def test_qrels_parse_and_keep_only_positives(tmp_path: Path) -> None:
    path = tmp_path / "q.txt"
    path.write_text("q1 0 d1 1\nq1 0 d2 0\nq2 0 d3 1\n", encoding="utf-8")
    gold = data.load_qrels(path)
    assert gold == {"q1": {"d1": 1}, "q2": {"d3": 1}}


def test_fixture_qrels_match_the_examples(domain_path: Path) -> None:
    gold = data.load_qrels(domain_path / "qrels_train.txt")
    examples = data.load_examples("iota", "train", domain_path)
    assert set(gold) == {e["id"] for e in examples}
    for example in examples:
        assert set(gold[example["id"]]) == set(example["gold_ids"])


# ------------------------------------------------------- restrict_to_corpus (§ subtle) --
def test_restrict_drops_unreachable_gold() -> None:
    gold = {"q1": {"real": 1, "ghost": 1}}
    assert data.restrict_to_corpus(gold, {"real"}) == {"q1": {"real": 1}}


def test_restrict_drops_topics_left_with_nothing() -> None:
    """This changes the denominator of the topic mean — num_topics < query count."""
    gold = {"q1": {"real": 1}, "q2": {"ghost": 1}}
    out = data.restrict_to_corpus(gold, {"real"})
    assert "q2" not in out
    assert len(out) == 1


def test_restrict_is_a_noop_when_all_gold_is_present() -> None:
    gold = {"q1": {"a": 1}, "q2": {"b": 1}}
    assert data.restrict_to_corpus(gold, {"a", "b"}) == gold


# --------------------------------------------------------------- ranking / tie-break --
def test_rank_documents_sorts_descending() -> None:
    ranked = runs.rank_documents(["a", "b", "c"], [0.1, 0.9, 0.5])
    assert [d for d, _ in ranked] == ["b", "c", "a"]


def test_ties_keep_corpus_order_not_alphabetical() -> None:
    """The organizers use a stable sort on score alone. Sorting by doc id would diverge."""
    ranked = runs.rank_documents(["z", "a"], [1.0, 1.0])
    assert [d for d, _ in ranked] == ["z", "a"]


def test_rank_documents_applies_top_k() -> None:
    ranked = runs.rank_documents(["a", "b", "c"], [3.0, 2.0, 1.0], top_k=2)
    assert len(ranked) == 2


def test_rank_documents_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="scores"):
        runs.rank_documents(["a", "b"], [1.0])


# ------------------------------------------------------------------ run round-trip --
def test_write_then_read_preserves_order(tmp_path: Path) -> None:
    ranked = {"q1": [("d2", 2.0), ("d1", 1.0)], "q2": [("d3", 5.0)]}
    path = runs.write_run(tmp_path / "run.trec", ranked)
    assert runs.read_run(path) == {"q1": ["d2", "d1"], "q2": ["d3"]}


def test_run_file_has_six_columns_and_one_based_ranks(tmp_path: Path) -> None:
    path = runs.write_run(tmp_path / "run.trec", {"q1": [("d1", 1.0), ("d2", 0.5)]}, tag="x")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert all(len(ln.split()) == 6 for ln in lines)
    assert lines[0].split()[3] == "1" and lines[1].split()[3] == "2"
    assert lines[0].split()[1] == "Q0"
    assert lines[0].split()[5] == "x"


def test_read_run_recovers_order_from_shuffled_lines(tmp_path: Path) -> None:
    """scorer.py sorts by (rank, -score), so line order in the file must not matter."""
    path = tmp_path / "run.trec"
    path.write_text("q1 Q0 d2 2 1.000000 t\nq1 Q0 d1 1 2.000000 t\n", encoding="utf-8")
    assert runs.read_run(path) == {"q1": ["d1", "d2"]}


def test_read_run_with_scores_round_trips(tmp_path: Path) -> None:
    path = runs.write_run(tmp_path / "run.trec", {"q1": [("d1", 0.25)]})
    assert runs.read_run_with_scores(path) == {"q1": {"d1": 0.25}}


def test_write_run_creates_missing_parents(tmp_path: Path) -> None:
    path = runs.write_run(tmp_path / "deep" / "nested" / "run.trec", {"q1": [("d1", 1.0)]})
    assert path.is_file()


def test_retriever_output_cannot_depend_on_gold(tmp_path) -> None:
    """§5.4: gold_ids must never reach inference.

    `reteco.data.topics_1a` attaches gold_ids to every Topic for convenience, and the
    Retriever signature takes those Topics — so a retriever *could* read them. An audit
    grep proves that no retriever does so today; this proves it stays true. Blanking the
    gold must not change a single ranking.
    """
    import dataclasses

    from reteco.data import Corpus, Topic
    from submit.make_runs import lexical_overlap_retriever

    corpus = Corpus(doc_ids=["d1", "d2", "d3"],
                    texts=["alpha beta", "beta gamma", "delta"], domain="t")
    with_gold = [Topic(topic_id="q1", text="beta", gold_ids=["d1", "d2"], domain="t"),
                 Topic(topic_id="q2", text="delta", gold_ids=["d3"], domain="t")]
    without = [dataclasses.replace(t, gold_ids=[]) for t in with_gold]

    assert lexical_overlap_retriever(corpus, with_gold, 10) == \
        lexical_overlap_retriever(corpus, without, 10), (
        "retrieval changed when gold was removed — it is reading the labels"
    )
