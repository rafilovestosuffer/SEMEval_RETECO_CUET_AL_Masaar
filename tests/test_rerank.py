"""Listwise reranking (Phase 7). The GPU is expensive; the bookkeeping must be right first."""

from __future__ import annotations

from reteco.rerank import (apply_permutation, build_messages, clean_response,
                           parse_permutation, replace_number, rerank_lists,
                           reranked_scores, window_plan)


def test_window_plan_matches_the_authors_loop() -> None:
    assert window_plan(30) == [(10, 30), (0, 20)]
    assert window_plan(20) == [(0, 20)]
    assert window_plan(100) == [(80, 100), (70, 90), (60, 80), (50, 70), (40, 60),
                                (30, 50), (20, 40), (10, 30), (0, 20)]
    assert window_plan(10) == [(0, 10)], "a list shorter than the window is one window"


def test_replace_number_protects_identifiers() -> None:
    assert replace_number("see [3] and [12]") == "see (3) and (12)"
    assert replace_number("[a] stays") == "[a] stays"


def test_build_messages_is_the_authors_prompt() -> None:
    msgs = build_messages("when [1] happened", ["alpha [2]", "beta"])
    assert msgs[0]["role"] == "system" and "<think>" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user.startswith("I will provide you with 2 passages")
    assert "query: when (1) happened.\n\n[1] alpha (2)\n[2] beta\nSearch Query:" in user
    assert user.endswith("e.g., [2] > [1].")


def test_clean_response_reads_only_the_answer_block() -> None:
    out = "<think>maybe [9] > [1]</think> <answer>[3] > [1] > [2]</answer>"
    assert clean_response(out) == "3     1     2"
    assert clean_response("<think>cut off before answering") is None
    assert clean_response("<think>x</think><answer>[2] > [1]").split() == ["2", "1"]


def test_parse_permutation_repairs_like_receive_permutation() -> None:
    perm, ok = parse_permutation("<think>t</think><answer>[3] > [3] > [9] > [1]</answer>", 4)
    assert ok and perm == [2, 0, 1, 3], "dupes and out-of-range dropped, omitted appended"
    perm, ok = parse_permutation("no answer at all", 3)
    assert not ok and perm == [0, 1, 2], "unparseable leaves the window unchanged"


def test_apply_permutation_touches_only_the_window() -> None:
    assert apply_permutation(list("abcde"), 1, 4, [2, 0, 1]) == list("adbce")


def test_rerank_lists_walks_windows_back_to_front_and_keeps_the_tail() -> None:
    docs = [f"d{i}" for i in range(40)]
    texts = {d: f"text of {d}" for d in docs}
    calls = []

    def reverse_everything(chats):
        calls.append(len(chats))
        outs = []
        for chat in chats:
            n = chat[1]["content"].count("\n[")
            order = " > ".join(f"[{i}]" for i in range(n, 0, -1))
            outs.append(f"<think>x</think><answer>{order}</answer>")
        return outs

    log: list = []
    out = rerank_lists({"q": "query"}, {"q": docs}, texts, reverse_everything, depth=30,
                       log=log)["q"]
    assert calls == [1, 1], "two windows for depth 30"
    # window (10,30) reversed puts d29..d10 at 10..29; window (0,20) then reverses
    # d0..d9 + d29..d20, so d20 leads and d0 sits at rank 20.
    assert out[0] == "d20" and out[19] == "d0"
    assert out[30:] == docs[30:], "below depth is never touched"
    assert sorted(out) == sorted(docs), "a permutation, nothing lost or duplicated"
    assert all(entry["parsed"] for entry in log) and len(log) == 2


def test_reranked_scores_are_strictly_decreasing() -> None:
    scores = [s for _, s in reranked_scores(["a", "b", "c"])]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 3
