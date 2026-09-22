#!/usr/bin/env python3
"""Compare two retrieval runs: scores, overlap, and where each one's gold actually is.

Built for the single question that decides Phase 5-7 spending: **does the weaker arm find
anything the stronger one misses?** The research pass converged on this from three
directions independently, and it is free to compute from runs we already have.

Reported per domain and overall:

- **nDCG@10 for each system and the paired delta.** The headline comparison.
- **recall@k for A, for B, and for their UNION.** The union is the ceiling a candidate-pool
  merge could reach; if it barely exceeds max(A, B), pooling has no headroom and fusion is
  not worth the complexity regardless of how the scores are combined.
- **Top-k overlap.** Low overlap is the usual argument *for* fusion, but it is not
  sufficient: two systems can disagree completely and one of them still be noise. Read it
  beside marginal gold, never alone.
- **Marginal gold** — judged documents A retrieves that B does not, and vice versa. This is
  the number that actually decides pooling, because it counts *relevant* disagreement
  rather than disagreement in general.

A caution the numbers cannot express: recall here is bounded by the run depth on disk.
Phase 2 and Phase 4c both write top-100, so ``--k`` above 100 silently measures nothing
extra. `recall@1000` needs runs regenerated at that depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reteco.data import load_qrels  # noqa: E402
from reteco.runs import read_run  # noqa: E402
from score import OFFICIAL_METRIC, per_topic_scores  # noqa: E402
from reteco.runs import read_run_with_scores  # noqa: E402

__all__ = ["overlap_at_k", "recall_at_k", "union_recall_at_k", "marginal_gold", "compare"]


def overlap_at_k(a: list[str], b: list[str], k: int) -> float:
    """Jaccard overlap of the two top-k document sets."""
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def union_recall_at_k(a: list[str], b: list[str], gold: set[str], k: int) -> float:
    """Recall of the merged candidate pool — the ceiling for any pooling strategy."""
    return len((set(a[:k]) | set(b[:k])) & gold) / len(gold) if gold else 0.0


def marginal_gold(a: list[str], b: list[str], gold: set[str], k: int) -> int:
    """Judged documents in A's top-k that are absent from B's top-k."""
    return len((set(a[:k]) & gold) - set(b[:k]))


def compare(runs_a: Path, runs_b: Path, qrels_dir: Path, subtrack: str, split: str,
            k: int = 100) -> dict:
    """Walk every domain present in both run sets and collect the comparison."""
    qrels_name = ("qrels_steps" if subtrack == "1b" else "qrels") + f"_{split}.txt"
    per_domain: dict[str, dict] = {}

    for domain_dir in sorted(p for p in Path(runs_a).iterdir() if p.is_dir()):
        domain = domain_dir.name
        fa = domain_dir / f"run_{subtrack}_{split}.trec"
        fb = Path(runs_b) / domain / f"run_{subtrack}_{split}.trec"
        fq = Path(qrels_dir) / domain / qrels_name
        if not (fa.is_file() and fb.is_file() and fq.is_file()):
            continue

        ra, rb = read_run(fa), read_run(fb)
        qrels = load_qrels(fq)
        sa = per_topic_scores(read_run_with_scores(fa), qrels)
        sb = per_topic_scores(read_run_with_scores(fb), qrels)

        rows = []
        for topic, judged in qrels.items():
            gold = set(judged)
            la, lb = ra.get(topic, []), rb.get(topic, [])
            if topic not in sa and topic not in sb:
                continue
            rows.append({
                "topic": topic,
                "ndcg_a": sa.get(topic, {}).get(OFFICIAL_METRIC, 0.0),
                "ndcg_b": sb.get(topic, {}).get(OFFICIAL_METRIC, 0.0),
                "recall_a": recall_at_k(la, gold, k),
                "recall_b": recall_at_k(lb, gold, k),
                "recall_union": union_recall_at_k(la, lb, gold, k),
                "overlap": overlap_at_k(la, lb, k),
                "marginal_a": marginal_gold(la, lb, gold, k),
                "marginal_b": marginal_gold(lb, la, gold, k),
            })
        if rows:
            per_domain[domain] = rows
    return per_domain


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def render(per_domain: dict, k: int, name_a: str, name_b: str) -> str:
    lines = [
        f"A = {name_a}    B = {name_b}    depth k={k}",
        "",
        f"  {'domain':<12}{'nDCG A':>8}{'nDCG B':>8}{'r@k A':>8}{'r@k B':>8}"
        f"{'r@k ∪':>8}{'overlap':>9}{'A-only':>8}{'B-only':>8}",
        f"  {'-'*12}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*9:>9}"
        f"{'-'*8:>8}{'-'*8:>8}",
    ]
    allrows: list[dict] = []
    for domain, rows in sorted(per_domain.items()):
        allrows += rows
        lines.append(
            f"  {domain:<12}{_mean(rows,'ndcg_a'):>8.4f}{_mean(rows,'ndcg_b'):>8.4f}"
            f"{_mean(rows,'recall_a'):>8.3f}{_mean(rows,'recall_b'):>8.3f}"
            f"{_mean(rows,'recall_union'):>8.3f}{_mean(rows,'overlap'):>9.3f}"
            f"{sum(r['marginal_a'] for r in rows):>8}{sum(r['marginal_b'] for r in rows):>8}")

    n = len(allrows)
    ra, rb, ru = (_mean(allrows, "recall_a"), _mean(allrows, "recall_b"),
                  _mean(allrows, "recall_union"))
    lines += [
        f"  {'-'*12}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*8:>8}{'-'*9:>9}"
        f"{'-'*8:>8}{'-'*8:>8}",
        f"  {'POOLED':<12}{_mean(allrows,'ndcg_a'):>8.4f}{_mean(allrows,'ndcg_b'):>8.4f}"
        f"{ra:>8.3f}{rb:>8.3f}{ru:>8.3f}{_mean(allrows,'overlap'):>9.3f}"
        f"{sum(r['marginal_a'] for r in allrows):>8}"
        f"{sum(r['marginal_b'] for r in allrows):>8}",
        "",
        f"  {n} topics. Pooled means are over queries — the leaderboard aggregation (§4).",
        "",
        f"  Union recall@{k} is {ru:.3f} against the better arm's {max(ra, rb):.3f}: "
        f"pooling could add at most **{ru - max(ra, rb):+.3f}**.",
    ]
    marg_b = sum(r["marginal_b"] for r in allrows)
    if ru - max(ra, rb) < 0.01:
        lines.append("  That headroom is negligible — pooling the two arms is not worth it, "
                     "whatever\n  operator is used. Spend the effort elsewhere.")
    else:
        lines.append(f"  B contributes {marg_b} judged documents A misses, so a pooled "
                     f"candidate set has\n  real headroom even if score fusion does not.")
    lines.append("\n  Overlap alone never justifies fusion: two systems can disagree "
                 "completely and one\n  still be noise. Read it beside the marginal-gold "
                 "columns.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="run dir for system A")
    parser.add_argument("--b", type=Path, required=True, help="run dir for system B")
    parser.add_argument("--qrels-dir", type=Path, default=Path("cache/qrels/track1_tempo"))
    parser.add_argument("--subtrack", choices=("1a", "1b"), default="1a")
    parser.add_argument("--split", default="train")
    parser.add_argument("--k", type=int, default=100,
                        help="depth; runs on disk are top-100, so >100 measures nothing extra")
    parser.add_argument("--json", type=Path, help="also write the per-topic table here")
    args = parser.parse_args()

    per_domain = compare(args.a, args.b, args.qrels_dir, args.subtrack, args.split, args.k)
    if not per_domain:
        print("no domains had all of: run A, run B, qrels", file=sys.stderr)
        return 1

    print(render(per_domain, args.k, args.a.name, args.b.name))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(per_domain, indent=0), encoding="utf-8")
        print(f"\nper-topic table -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
