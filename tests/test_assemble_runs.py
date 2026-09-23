"""The submission assembler: a topic the reranker did not reach must fall back, never vanish."""

from __future__ import annotations

import json
from pathlib import Path

from submit.assemble_runs import expected_topics, merge_1a


def test_merge_prefers_reranked_and_falls_back_to_fused() -> None:
    fused = {"q1": {"a": 0.9, "b": 0.8}, "q2": {"c": 0.7, "d": 0.6}}
    reranked = {"q1": {"b": 2.0, "a": 1.0}}
    run, used = merge_1a(fused, reranked)
    assert used == 1
    assert [d for d, _ in run["q1"]] == ["b", "a"], "reranked order wins"
    assert [d for d, _ in run["q2"]] == ["c", "d"], "unreached topic keeps the fused order"
    assert set(run) == {"q1", "q2"}, "no topic dropped"


def test_merge_ignores_reranked_topics_outside_the_fused_run() -> None:
    run, used = merge_1a({"q1": {"a": 1.0}}, {"stray": {"x": 1.0}})
    assert set(run) == {"q1"} and used == 0


def test_expected_topics_reads_query_files_not_qrels(tmp_path: Path) -> None:
    (tmp_path / "examples_dev.jsonl").write_text(
        json.dumps({"id": "1_2", "query": "q"}) + "\n", encoding="utf-8")
    (tmp_path / "steps_dev.jsonl").write_text(json.dumps(
        {"id": "1_2", "query": "q", "steps": [{"step_id": "1_2_step1"},
                                              {"step_id": "1_2_step2"}]}) + "\n",
        encoding="utf-8")
    assert expected_topics(tmp_path, "dev") == {"1a": {"1_2"}, "1b": {"1_2_step1", "1_2_step2"}}
    assert expected_topics(tmp_path, "test") == {"1a": set(), "1b": set()}
