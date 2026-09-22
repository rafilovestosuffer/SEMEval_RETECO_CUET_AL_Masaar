"""Listwise LLM reranking with ReasonRank (Phase 7, H4).

Why this reranker (`notes/lit/reasonrank.md`): on BRIGHT, reranking ReasonIR's top-100
(30.59 nDCG@10), almost every reranker under 32B makes the list *worse* — RankT5 16.60,
RankZephyr 22.64, Rank1-7B 27.23, Rank-R1-7B 20.57. Only Rearank-7B (31.75) and
ReasonRank-7B (35.74) improve on the retriever. That matches our own prior that MS
MARCO-lineage rerankers are presumed harmful here (`notes/lit/SUMMARY.md` row 7).

Everything that shapes the model input is ported **verbatim** from the authors' inference
code (github.com/8421BCD/ReasonRank @ 15d8733, ``utils.py``, ``rerank/rankllm.py``,
``rerank/rank_listwise_os_llm.py``, ``listwise_prompt_r1.toml``), because a listwise
reranker trained with one prompt is not guaranteed to work with a paraphrase of it:

- the ``rank_GPT_reasoning`` system prompt, prefix and suffix;
- ``[n]`` in queries and passages rewritten to ``(n)`` so they cannot collide with the
  passage identifiers;
- passages truncated to 512 *tokenizer* tokens (their BRIGHT setting);
- sliding windows walked back to front, window 20, step 10;
- the permutation parser, including its fallback: identifiers the model omits keep their
  original relative order after the ones it ranked, and an unparseable answer leaves the
  window unchanged.

This module is pure Python: the tokenizer and the generator are passed in, so the whole
pipeline is unit-tested on CPU with a fake generator before any GPU time is spent.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

__all__ = ["SYSTEM_PROMPT", "ANSWER_PATTERN", "replace_number", "prefix_prompt",
           "post_prompt", "build_messages", "clean_response", "parse_permutation",
           "window_plan", "apply_permutation", "rerank_lists", "reranked_scores"]

SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank passages based on their "
    "relevance to the query. Given a query and a passage list, you first thinks about the "
    "reasoning process in the mind and then provides the answer (i.e., the reranked passage "
    "list). The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)
ANSWER_PATTERN = r"<think>.*?</think>\s*<answer>(.*?)</answer>"

Messages = list[dict[str, str]]
Generate = Callable[[list[Messages]], list[str]]


def replace_number(s: str) -> str:
    """``[3]`` -> ``(3)``, so text cannot impersonate a passage identifier."""
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


def prefix_prompt(query: str, num: int) -> str:
    return (f"I will provide you with {num} passages, each indicated by a numerical "
            f"identifier []. Rank the passages based on their relevance to the search "
            f"query: {query}.\n")


def post_prompt(query: str, num: int) -> str:
    return (f"Search Query: {query}.\nRank the {num} passages above based on their relevance "
            f"to the search query. All the passages should be included and listed using "
            f"identifiers, in descending order of relevance. The format of the answer should "
            f"be [] > [], e.g., [2] > [1].")


def build_messages(query: str, passages: Sequence[str],
                   prepare: Callable[[str], str] = lambda s: s) -> Messages:
    """The chat for one window. ``prepare`` is fix_text + 512-token truncation."""
    query = replace_number(query).strip()
    num = len(passages)
    body = f"{prefix_prompt(query, num)}\n"
    for rank, text in enumerate(passages, start=1):
        body += f"[{rank}] {replace_number(prepare(text))}\n"
    body += post_prompt(query, num)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body}]


def clean_response(response: str) -> str | None:
    """Digits of the ``<answer>`` block, space-separated; None when there is no answer.

    The authors match the pattern against ``response.lower()``, and fall back to whatever
    follows the last ``<answer>`` when the closing tag is missing (reasoning cut off).
    """
    match = re.search(ANSWER_PATTERN, response.lower(), re.DOTALL)
    if match:
        response = match.group(1).strip()
    elif "<answer>" in response:
        response = response.split("<answer>")[-1]
    else:
        return None
    return "".join(c if c.isdigit() else " " for c in response).strip()


def parse_permutation(response: str, num: int) -> tuple[list[int], bool]:
    """0-based permutation of ``range(num)``, and whether the model's answer was usable.

    Out-of-range ids and duplicates are dropped; ids the model omitted are appended in
    their original order — exactly the authors' ``receive_permutation``.
    """
    cleaned = clean_response(response)
    if not cleaned:
        return list(range(num)), False
    seen: list[int] = []
    for token in cleaned.split():
        idx = int(token) - 1
        if 0 <= idx < num and idx not in seen:
            seen.append(idx)
    if not seen:
        return list(range(num)), False
    return seen + [i for i in range(num) if i not in seen], True


def window_plan(depth: int, window: int = 20, step: int = 10) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` in execution order, back to front — the authors' loop.

    depth 30 -> [(10, 30), (0, 20)]; depth 20 -> [(0, 20)]; depth 100 -> 9 windows.
    A depth at or below the window is one window: the loop alone would yield nothing, and
    the authors route such short lists through a separate single-window path.
    """
    if depth <= window:
        return [(0, depth)]
    rank_start, end, start = 0, depth, depth - window
    plan = []
    while end > rank_start and start + step != rank_start:
        start = max(start, rank_start)
        plan.append((start, end))
        end -= step
        start -= step
    return plan


def apply_permutation(items: list, start: int, end: int, perm: Sequence[int]) -> list:
    window = items[start:end]
    return items[:start] + [window[i] for i in perm] + items[end:]


def rerank_lists(queries: dict[str, str], candidates: dict[str, list[str]],
                 texts: dict[str, str], generate: Generate, depth: int = 30,
                 window: int = 20, step: int = 10,
                 prepare: Callable[[str], str] = lambda s: s,
                 log: list | None = None) -> dict[str, list[str]]:
    """Rerank the top ``depth`` of every list; everything below ``depth`` is untouched.

    Windows are batched *across queries*: every query's first window goes to the
    generator in one call, then every query's second window. That is what lets vLLM
    keep the GPU full, and it is the order the authors use (``sliding_windows_batched``).

    Args:
        candidates: ``{topic: [doc_id, ...]}`` in first-stage order.
        texts: ``{doc_id: content}`` covering at least the top ``depth`` of every list.
        log: if given, one dict per generated window is appended (topic, window, raw
            output, parsed flag) — raw outputs are kept so parsing can be audited later.
    """
    lists = {t: list(c) for t, c in candidates.items() if t in queries}
    for start, end in window_plan(depth, window, step):
        batch = [t for t, c in lists.items() if len(c) > start + 1]
        chats = [build_messages(queries[t], [texts[d] for d in lists[t][start:end]], prepare)
                 for t in batch]
        outputs = generate(chats) if chats else []
        for topic, out in zip(batch, outputs):
            n = len(lists[topic][start:end])
            perm, ok = parse_permutation(out, n)
            lists[topic] = apply_permutation(lists[topic], start, end, perm)
            if log is not None:
                log.append({"topic": topic, "window": [start, end], "parsed": ok,
                            "output": out})
    return lists


def reranked_scores(ranked: Sequence[str]) -> list[tuple[str, float]]:
    """Strictly decreasing scores that encode the final order, for a TREC run."""
    n = len(ranked)
    return [(doc, float(n - i)) for i, doc in enumerate(ranked)]
