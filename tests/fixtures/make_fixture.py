#!/usr/bin/env python3
"""Generate a synthetic Track 1 domain in the real RETECO release schema.

Usage::

    python tests/fixtures/make_fixture.py --out /tmp/fake_reteco --docs 300 --queries 12

Produces ``<out>/track1_tempo/<domain>/`` containing ``documents.jsonl``,
``examples_{train,dev}.jsonl``, ``steps_{train,dev}.jsonl`` and ``qrels_*.txt`` —
the exact filenames and field names ``official_baseline.py`` reads.

This exists to exercise the pyserini + gensim + pytrec_eval toolchain end to end
without the real corpus, which a Claude Code session cannot download
(huggingface.co is blocked). It proves the stack RUNS. It proves nothing about
whether any nDCG value is correct — only real IOTA does that.

Deliberately NOT the starter kit's docs/sample_data: that ships a flattened
steps.jsonl with no ``steps[]`` array, and a documents.jsonl holding only gold
passages, which its own README says must not be used for retrieval.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20260916

TOPICS = ["consensus", "ledger", "throughput", "wallet", "oracle", "sharding",
          "governance", "staking", "latency", "fork", "mempool", "signature"]
YEARS = [2017, 2019, 2020, 2021, 2022, 2023, 2024]
FILLER = ("network protocol node transaction block validator token upgrade release "
          "proposal community developer implementation performance").split()


def build(out: Path, domain: str, n_docs: int, n_queries: int,
          gold_per_query: int, steps_per_query: int) -> Path:
    rng = random.Random(SEED)
    root = out / "track1_tempo" / domain
    root.mkdir(parents=True, exist_ok=True)

    # ---- corpus. Doc ids mirror the real format: '<domain>/<hash>_<n>.txt'.
    docs = []
    for i in range(n_docs):
        topic = rng.choice(TOPICS)
        year = rng.choice(YEARS)
        body = " ".join(rng.choice(FILLER) for _ in range(rng.randint(40, 90)))
        docs.append({
            "id": f"{domain}/{rng.getrandbits(32):08x}_{i}.txt",
            "content": f"In {year} the {topic} was revised. {body} "
                       f"This concerns {topic} during {year}.",
        })
    (root / "documents.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")

    # ---- queries, split 70/30 like the real release.
    n_train = max(1, round(n_queries * 0.7))
    splits = {"train": range(n_train), "dev": range(n_train, n_queries)}

    for split, indices in splits.items():
        examples, steps, qrels, step_qrels = [], [], [], []
        for qi in indices:
            topic = rng.choice(TOPICS)
            year = rng.choice(YEARS)
            qid = f"{rng.getrandbits(24)}_{qi}"
            gold = [d["id"] for d in rng.sample(docs, gold_per_query)]

            examples.append({
                "id": qid,
                "query": f"What changed about {topic} in {year}?",
                "gold_ids": gold,
                "gold_answers": [f"The {topic} changed in {year}."],
            })
            qrels += [f"{qid} 0 {g} 1" for g in gold]

            # steps_*.jsonl nests steps beneath the query -- topics_1b reads
            # record['steps'] and each step's 'step_id' / 'step_instruction'.
            step_records = []
            for si in range(1, steps_per_query + 1):
                step_id = f"{qid}_step{si}"
                step_gold = rng.sample(gold, max(1, gold_per_query // steps_per_query))
                step_records.append({
                    "step_id": step_id,
                    "step": f"Find {topic} state in {year - si}",
                    "step_instruction": f"Retrieve passages describing {topic} "
                                        f"as of {year - si}.",
                    "gold_ids": step_gold,
                })
                step_qrels += [f"{step_id} 0 {g} 1" for g in step_gold]
            steps.append({"id": qid,
                          "query": f"What changed about {topic} in {year}?",
                          "steps": step_records})

        (root / f"examples_{split}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
        (root / f"steps_{split}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
        (root / f"qrels_{split}.txt").write_text("\n".join(qrels) + "\n", encoding="utf-8")
        (root / f"qrels_steps_{split}.txt").write_text(
            "\n".join(step_qrels) + "\n", encoding="utf-8")

    (root / "split_manifest.json").write_text(json.dumps(
        {"domain": domain, "documents": n_docs, "queries": n_queries,
         "train": n_train, "dev": n_queries - n_train, "seed": SEED,
         "synthetic": True}, indent=2) + "\n", encoding="utf-8")

    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", default="iota")
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--gold-per-query", type=int, default=4)
    parser.add_argument("--steps-per-query", type=int, default=2)
    args = parser.parse_args()

    root = build(args.out, args.domain, args.docs, args.queries,
                 args.gold_per_query, args.steps_per_query)
    print(f"wrote synthetic domain to {root}")
    for path in sorted(root.iterdir()):
        print(f"  {path.name:<28} {path.stat().st_size:>8} bytes")
    print("\nSYNTHETIC DATA — for toolchain smoke-testing only. Any nDCG computed from")
    print("this is meaningless; only the real IOTA domain can close the Phase 1 gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
