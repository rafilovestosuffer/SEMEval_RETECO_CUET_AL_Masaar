"""Loaders for the RETECO Track 1 release schema.

Field names and semantics are taken from the organizers' `official_baseline.py` at commit
`23093c3`, recorded in `notes/starter_kit_findings.md` §3. Where their code and their prose
disagree, the code wins.

Per-domain layout (CLAUDE.md §4)::

    reteco_data/track1_tempo/<domain>/
        documents.jsonl          {"id": ..., "content": ...}
        examples_{train,dev}.jsonl   {"id", "query", "gold_ids", "gold_answers"}
        steps_{train,dev}.jsonl      {"id", "query", "steps": [{"step_id", "step",
                                      "step_instruction", "gold_ids"}]}
        qrels_{train,dev}.txt        "qid 0 docid rel", space separated, all rel = 1
        qrels_steps_{train,dev}.txt
        guidance_*.jsonl
        split_manifest.json

Two behaviours here are easy to get subtly wrong and both change the reported score:
`restrict_to_corpus` (see its docstring) and the 1b query template, which uses
`step_instruction` and **not** `step`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from reteco.paths import domain_dir

__all__ = [
    "SPLITS",
    "Corpus",
    "Topic",
    "read_jsonl",
    "load_corpus",
    "load_examples",
    "load_steps",
    "load_qrels",
    "topics_1a",
    "topics_1b",
    "restrict_to_corpus",
    "build_1b_query",
    "available_domains",
]

SPLITS = ("train", "dev")

# Tempo/run_step.py, via official_baseline.py::topics_1b. The docstring there says
# "step_text"; the code uses step_instruction. The code is what produced the published
# numbers, so this template is authoritative.
STEP_QUERY_TEMPLATE = "{query}\n\nStep: {step_instruction}"


def build_1b_query(query: str, step_instruction: str) -> str:
    """The official sub-track 1b query: base query, blank line, ``Step: <instruction>``."""
    return STEP_QUERY_TEMPLATE.format(query=query, step_instruction=step_instruction)


@dataclass(frozen=True)
class Corpus:
    """One domain's document collection, in file order.

    Order matters: the organizers break score ties with a stable sort, so two documents with
    equal scores rank by their position in ``documents.jsonl`` (never by doc id).
    """

    doc_ids: list[str]
    texts: list[str]
    domain: str = ""

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def id_set(self) -> set[str]:
        return set(self.doc_ids)

    def position(self) -> dict[str, int]:
        """doc_id -> index in file order, for reproducing the official tie-break."""
        return {doc_id: i for i, doc_id in enumerate(self.doc_ids)}


@dataclass(frozen=True)
class Topic:
    """One scored unit: a 1a query or a 1b step."""

    topic_id: str
    text: str
    gold_ids: list[str] = field(default_factory=list)
    domain: str = ""
    parent_id: str = ""  # for 1b, the query id the step belongs to; empty for 1a


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed object per non-blank line."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_corpus(domain: str, key: str = "id", root: Path | None = None) -> Corpus:
    """Load ``documents.jsonl`` for a domain, preserving file order.

    Args:
        domain: e.g. ``"iota"``.
        key: doc-id field — ``"id"`` for Track 1, ``"doc_id"`` for Track 2.
        root: override the domain directory (default: resolved via ``reteco.paths``).
    """
    directory = root if root is not None else domain_dir(domain)
    doc_ids, texts = [], []
    for record in read_jsonl(Path(directory) / "documents.jsonl"):
        doc_ids.append(record[key])
        texts.append(record["content"])
    return Corpus(doc_ids=doc_ids, texts=texts, domain=domain)


def load_examples(domain: str, split: str, root: Path | None = None) -> list[dict]:
    """Raw records from ``examples_{split}.jsonl``."""
    directory = root if root is not None else domain_dir(domain)
    return list(read_jsonl(Path(directory) / f"examples_{split}.jsonl"))


def load_steps(domain: str, split: str, root: Path | None = None) -> list[dict]:
    """Raw records from ``steps_{split}.jsonl`` (each has a nested ``steps`` list)."""
    directory = root if root is not None else domain_dir(domain)
    return list(read_jsonl(Path(directory) / f"steps_{split}.jsonl"))


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a TREC qrels file into ``{topic_id: {doc_id: rel}}``.

    Only positive judgments are kept, matching the organizers' `scorer.py::load_qrels`.
    Track 1 qrels are binary — every positive is grade 1.
    """
    gold: dict[str, dict[str, int]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            topic_id, _, doc_id, rel = line.split()
            if int(rel) > 0:
                gold[topic_id][doc_id] = int(rel)
    return dict(gold)


def topics_1a(domain: str, split: str, root: Path | None = None) -> list[Topic]:
    """Sub-track 1a topics: one per query, id = ``record["id"]``, text = the raw query."""
    return [
        Topic(topic_id=r["id"], text=r["query"], gold_ids=list(r.get("gold_ids", [])),
              domain=domain)
        for r in load_examples(domain, split, root)
    ]


def topics_1b(domain: str, split: str, root: Path | None = None) -> list[Topic]:
    """Sub-track 1b topics: one per step, id = ``step["step_id"]``.

    Text is the official template — base query + ``Step: <step_instruction>``. Note this is
    TEMPO's best-performing variant (Query+Step, 26.4 avg nDCG@10); retrieving with the step
    alone scores 14.6 and must not be used (`notes/lit/tempo.md`).
    """
    topics = []
    for record in load_steps(domain, split, root):
        for step in record.get("steps", []):
            topics.append(Topic(
                topic_id=step["step_id"],
                text=build_1b_query(record["query"], step["step_instruction"]),
                gold_ids=list(step.get("gold_ids", [])),
                domain=domain,
                parent_id=record["id"],
            ))
    return topics


def restrict_to_corpus(gold: dict[str, dict[str, int]],
                       corpus_ids: set[str]) -> dict[str, dict[str, int]]:
    """Drop unreachable judgments, then drop topics left with nothing.

    Verbatim semantics of `official_baseline.py::restrict_to_corpus`: gold ids naming a
    document absent from the corpus are removed, and a topic whose judgments are *all*
    removed disappears entirely.

    This is not cosmetic. The official score is a mean over surviving topics, so dropping a
    topic changes the denominator — `num_topics` can be smaller than the query count, and a
    reimplementation that keeps empty topics will silently report a different number.
    """
    out = {}
    for topic_id, docs in gold.items():
        keep = {d: r for d, r in docs.items() if d in corpus_ids}
        if keep:
            out[topic_id] = keep
    return out


def available_domains(root: Path | None = None) -> list[str]:
    """Domain directory names present under the Track 1 root, sorted."""
    from reteco.paths import data_root, TRACK1_PREFIX

    base = Path(root) if root is not None else data_root() / TRACK1_PREFIX
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())
