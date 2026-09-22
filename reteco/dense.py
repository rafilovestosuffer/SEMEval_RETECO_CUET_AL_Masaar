"""Dense retrieval over the embeddings produced by Phase 4b.

Phase 4b stores, per domain, an fp16 matrix of **unique** document vectors plus a
`doc_index.json` mapping every document id to its row. That deduplication (29.4% of the
corpus) is invisible to scoring, and this module keeps it that way — but it also exploits
it: similarity is computed **once per unique row**, then read off for each document id.
For History, where 43.7% of documents are duplicates, that nearly halves the matmul.

Two details that decide whether this matches the official baseline's behaviour:

- **Scores are dot products on L2-normalised vectors**, i.e. cosine, which is what the
  model card specifies. Phase 4b normalises before writing, so nothing is renormalised
  here; a vector that is not unit-norm indicates a corrupt shard and is worth catching.
- **Ties break by corpus order.** Duplicate documents get *identical* scores by
  construction, so ties are common here in a way they never were for BM25 — every
  duplicate group is an exact tie. `reteco.runs.rank_documents` reproduces the organizers'
  stable sort, so ranking goes through it rather than being re-implemented.

Queries must be encoded with the model's **query prompt** and documents without it; see
`QUERY_PROMPT`. Getting that backwards puts the two sides in different regions of the
space and degrades silently rather than failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reteco.runs import TOP_K, rank_documents

__all__ = ["QUERY_PROMPT", "DomainIndex", "load_domain", "search", "search_all"]

# config_sentence_transformers.json @ AQ-MedAI/Diver-Retriever-0.6B, read 2026-09-22.
# Documents use the empty prompt; only queries carry this.
QUERY_PROMPT = ("Instruct: Given a web search query, retrieve relevant passages that "
                "answer the query\nQuery:")


@dataclass(frozen=True)
class DomainIndex:
    """One domain's dense index: unique vectors plus the document -> row mapping."""

    vectors: np.ndarray          # [n_unique, dim], float32, L2-normalised
    doc_ids: list[str]           # every document, in corpus order
    rows: np.ndarray             # [n_docs] int32, index into ``vectors``

    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def n_unique(self) -> int:
        return int(self.vectors.shape[0])


def load_domain(directory: Path, check_norms: bool = True) -> DomainIndex:
    """Load one domain's shards and index, concatenating shards in numeric order.

    Args:
        directory: holds ``emb_000.npy`` ... and ``doc_index.json``.
        check_norms: verify vectors are unit-norm. Phase 4b writes them normalised, so a
            failure here means a truncated or corrupt shard — worth catching before it
            silently distorts every score.

    Raises:
        FileNotFoundError: no shards, or no doc_index.json.
        ValueError: shard rows do not sum to ``n_unique``, or norms are wrong.
    """
    directory = Path(directory)
    index_path = directory / "doc_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no doc_index.json in {directory}")
    meta = json.loads(index_path.read_text(encoding="utf-8"))

    shards = sorted(directory.glob("emb_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no emb_*.npy shards in {directory}")
    vectors = np.concatenate([np.load(s) for s in shards], axis=0).astype(np.float32)

    expected = int(meta["n_unique"])
    if vectors.shape[0] != expected:
        raise ValueError(f"{directory.name}: {vectors.shape[0]} vectors across "
                         f"{len(shards)} shards but doc_index says {expected}")

    if check_norms:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-2):
            raise ValueError(f"{directory.name}: vectors are not unit-norm "
                             f"(min {norms.min():.4f}, max {norms.max():.4f})")

    return DomainIndex(vectors=vectors,
                       doc_ids=list(meta["doc_ids"]),
                       rows=np.asarray(meta["rows"], dtype=np.int32))


def search(index: DomainIndex, query_vectors: np.ndarray, query_ids: list[str],
           top_k: int = TOP_K,
           block: int = 64) -> dict[str, list[tuple[str, float]]]:
    """Rank documents for each query by cosine similarity.

    Similarity is computed against the **unique** vectors and then expanded to document
    ids, so duplicate documents cost nothing extra and receive identical scores.

    Args:
        query_vectors: [n_queries, dim]; L2-normalised here if they are not already.
        block: queries per matmul. Bounds peak memory at ``block x n_unique`` floats —
            History's 200k unique vectors at block 64 is ~50 MB, which is the point.

            **A determinism caveat.** Changing ``block`` changes the matmul shape, and BLAS
            picks different accumulation orders for different shapes, so scores move in the
            last couple of float32 bits (~1e-7 measured). Rankings are stable in practice
            and exact duplicates are unaffected, since they read the same row — but a run
            file is not bit-reproducible across block sizes, and two genuinely distinct
            documents whose true scores differ by less than that could swap. Keep ``block``
            fixed when comparing two systems, and prefer the paired bootstrap over
            eyeballing score differences (§9).

    Returns:
        ``{query_id: [(doc_id, score), ...]}``, at most ``top_k`` per query, ties broken
        by corpus order via `reteco.runs.rank_documents`.
    """
    if len(query_ids) != query_vectors.shape[0]:
        raise ValueError(f"{len(query_ids)} query ids but "
                         f"{query_vectors.shape[0]} query vectors")
    if query_vectors.shape[1] != index.vectors.shape[1]:
        raise ValueError(f"query dim {query_vectors.shape[1]} != "
                         f"index dim {index.vectors.shape[1]}")

    queries = np.asarray(query_vectors, dtype=np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.maximum(norms, 1e-12)

    out: dict[str, list[tuple[str, float]]] = {}
    for start in range(0, len(query_ids), block):
        chunk = queries[start:start + block]
        sims = chunk @ index.vectors.T              # [b, n_unique]
        per_doc = sims[:, index.rows]               # [b, n_docs] — duplicates share a score
        for i, qid in enumerate(query_ids[start:start + block]):
            out[qid] = rank_documents(index.doc_ids, per_doc[i].tolist(), top_k=top_k)
    return out


def search_all(root: Path, queries_by_domain: dict[str, tuple[list[str], np.ndarray]],
               top_k: int = TOP_K) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Run `search` for every domain that has both an index and queries.

    Domains are loaded one at a time and released, because holding all 13 indexes at once
    is ~3.4 GB and there is no reason to.
    """
    results: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for domain, (qids, qvecs) in sorted(queries_by_domain.items()):
        directory = Path(root) / domain
        if not (directory / "doc_index.json").is_file():
            continue
        index = load_domain(directory)
        results[domain] = search(index, qvecs, qids, top_k=top_k)
        del index
    return results
