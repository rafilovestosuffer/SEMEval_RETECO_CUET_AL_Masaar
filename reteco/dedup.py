"""Deduplicate a corpus by exact content, and order it for efficient encoding.

Two measured facts about this corpus drive the whole module (`notes/data_audit.md`):

1. **29.4% of documents are byte-identical duplicates** — 485,683 of 1,654,055, with
   History at 43.7% and bitcoin at 49.9%. Zero duplicate *ids* and zero empty documents,
   so it is many ids pointing at the same text. Embedding is the dominant GPU cost of the
   project, so encoding each distinct text once is a ~29% saving on it, for free and with
   no effect on the ranking: identical text must produce an identical vector anyway.
2. **The mean document is 158 tokens against a 512 cap**, so padding every batch to the
   cap wastes ~3.2x. Sorting by length before batching removes almost all of that, and it
   costs nothing in quality.

Both are pure bookkeeping — they change what the GPU is asked to do, never what it
computes. That matters for the paper: neither is an approximation to disclose as a
deviation, unlike truncation or chunking.

The scatter step is the part that is easy to get wrong. After encoding `unique_texts`, a
document's vector is ``vectors[doc_to_unique[doc_id]]``; every duplicate resolves to the
same row rather than being dropped, so the run file still contains all 1.65M documents and
the ranking is unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

__all__ = ["content_key", "DedupPlan", "build_plan"]


def content_key(text: str) -> str:
    """Stable key for exact-duplicate detection.

    BLAKE2b-128 over the raw UTF-8 bytes. Exact, not fuzzy: the audit measured
    byte-identical duplication, and near-duplicate clustering would be a modelling
    decision that changes results and would need its own ablation. No normalisation
    (case, whitespace) for the same reason — two texts differing only in whitespace are
    different inputs to the encoder and could legitimately embed differently.
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class DedupPlan:
    """What to encode, in what order, and how to map results back to documents."""

    unique_texts: list[str]
    """Distinct texts, ordered ascending by token-length proxy for batching."""

    doc_to_unique: dict[str, int]
    """doc_id -> index into ``unique_texts`` (and so into the encoded matrix)."""

    duplicates: int = 0
    """How many documents resolved to a text already seen."""

    order_proxy: list[int] = field(default_factory=list)
    """Length proxy per unique text, in the same order — for shard sizing."""

    def __len__(self) -> int:
        return len(self.unique_texts)

    @property
    def saving(self) -> float:
        """Fraction of encode work avoided, by document count."""
        total = len(self.doc_to_unique)
        return self.duplicates / total if total else 0.0


def build_plan(doc_ids: list[str], texts: list[str],
               sort_by_length: bool = True) -> DedupPlan:
    """Map documents onto the distinct texts that need encoding.

    Args:
        doc_ids: ids in corpus order.
        texts: the matching document contents.
        sort_by_length: order unique texts shortest-first so batches pad to near the
            longest member rather than to ``max_length``. Pass False to keep first-seen
            order, which makes the mapping easier to eyeball in tests.

    The length proxy is ``len(text)`` in characters, not tokens: tokenising 1.65M
    documents twice to save a sort is not worth it, and characters and tokens are
    monotonically related closely enough for batching.

    Raises:
        ValueError: if the two lists differ in length, which would silently misalign
            every id with the wrong text.
    """
    if len(doc_ids) != len(texts):
        raise ValueError(f"{len(doc_ids)} doc_ids but {len(texts)} texts")

    first_index: dict[str, int] = {}
    unique: list[str] = []
    assigned: list[tuple[str, int]] = []
    duplicates = 0

    for doc_id, text in zip(doc_ids, texts):
        key = content_key(text)
        index = first_index.get(key)
        if index is None:
            index = len(unique)
            first_index[key] = index
            unique.append(text)
        else:
            duplicates += 1
        assigned.append((doc_id, index))

    if not sort_by_length:
        return DedupPlan(unique_texts=unique, doc_to_unique=dict(assigned),
                         duplicates=duplicates,
                         order_proxy=[len(t) for t in unique])

    # Reorder unique texts shortest-first, then rewrite every index through the permutation.
    order = sorted(range(len(unique)), key=lambda i: len(unique[i]))
    remap = {old: new for new, old in enumerate(order)}
    sorted_texts = [unique[i] for i in order]
    return DedupPlan(
        unique_texts=sorted_texts,
        doc_to_unique={doc_id: remap[i] for doc_id, i in assigned},
        duplicates=duplicates,
        order_proxy=[len(t) for t in sorted_texts],
    )
