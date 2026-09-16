#!/usr/bin/env python3
"""Phase 1 gate: does our BM25 run reproduce the organizers' published IOTA numbers?

Usage::

    python eval/gate.py --results cache/kernel_output/reteco-phase1-bm25-iota/results.json
    python eval/gate.py --results results.json --domain iota --tolerance 1e-4
    python eval/gate.py --expected            # just print the published table

Reads the per-domain ``results.json`` written by the organizers' ``official_baseline.py``
(keys like ``1a_train`` -> ``{"NDCG@10": ..., "num_topics": ...}``) and compares its
nDCG@10 values against the figures published in the starter kit's ``BASELINE_RESULTS.md``
at commit 23093c3.

Exit code 0 = gate passed, 1 = failed (CLAUDE.md §9, Phase 1).

A note on what this can and cannot prove — see ``LOAD_BEARING`` below. IOTA 1b train is
published as 0.0000, so matching it is not evidence of anything: a completely broken
pipeline returns 0.0000 too. It is reported but does not count toward the verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reteco.paths import cache_root  # noqa: E402

__all__ = ["PUBLISHED", "LOAD_BEARING", "GateRow", "compare", "verdict"]

# Transcribed from starter_kit/BASELINE_RESULTS.md @ 23093c3, "Per-domain nDCG@10",
# Track 1 · TEMPO. These are the ORGANIZERS' published figures, not a measurement of
# ours -- nothing here belongs in results/ledger.csv (CLAUDE.md §5.6).
PUBLISHED: dict[str, dict[str, float]] = {
    "iota": {"1a_train": 0.0199, "1a_dev": 0.2083, "1b_train": 0.0000, "1b_dev": 0.3289},
    "bitcoin": {"1a_train": 0.0695, "1a_dev": 0.0263, "1b_train": 0.0774, "1b_dev": 0.0205},
    "cardano": {"1a_train": 0.1349, "1a_dev": 0.0851, "1b_train": 0.1174, "1b_dev": 0.0554},
    "economics": {"1a_train": 0.0382, "1a_dev": 0.0480, "1b_train": 0.0278, "1b_dev": 0.0517},
    "genealogy": {"1a_train": 0.1003, "1a_dev": 0.1677, "1b_train": 0.0982, "1b_dev": 0.2060},
    "history": {"1a_train": 0.0691, "1a_dev": 0.0877, "1b_train": 0.0651, "1b_dev": 0.0989},
    "hsm": {"1a_train": 0.1627, "1a_dev": 0.2239, "1b_train": 0.1591, "1b_dev": 0.2084},
    "law": {"1a_train": 0.0943, "1a_dev": 0.0574, "1b_train": 0.0846, "1b_dev": 0.0549},
    "monero": {"1a_train": 0.0278, "1a_dev": 0.0252, "1b_train": 0.0517, "1b_dev": 0.0103},
    "politics": {"1a_train": 0.2792, "1a_dev": 0.2550, "1b_train": 0.2425, "1b_dev": 0.2306},
    "quant": {"1a_train": 0.0255, "1a_dev": 0.0218, "1b_train": 0.0085, "1b_dev": 0.0374},
    "travel": {"1a_train": 0.0429, "1a_dev": 0.0275, "1b_train": 0.0378, "1b_dev": 0.0492},
    "workplace": {"1a_train": 0.0777, "1a_dev": 0.0230, "1b_train": 0.1369, "1b_dev": 0.0302},
}

# Published macro-averages over the 13 domains, for Phase 2's gate.
PUBLISHED_MACRO = {"1a_train": 0.0879, "1a_dev": 0.0967, "1b_train": 0.0852, "1b_dev": 0.1063}

# A published 0.0000 is not evidence -- a broken pipeline reproduces it by accident.
# Only keys whose published value is non-zero count toward the verdict.
NDCG_KEY = "NDCG@10"

# CLAUDE.md §9 asks for a match "to 4 decimals". The published figures are already
# rounded to 4 dp, so the correct test is that the observed value ROUNDS to the
# published one -- i.e. a strict half-ulp bound. A tolerance of 1e-4 would be wrong:
# it admits a full unit in the 4th decimal, letting 0.0200 "match" 0.0199.
DEFAULT_TOLERANCE = 5e-5


def load_bearing(domain: str) -> list[str]:
    """Sub-track/split keys whose published value for ``domain`` is non-zero."""
    return [k for k, v in sorted(PUBLISHED[domain].items()) if v != 0.0]


LOAD_BEARING = {d: load_bearing(d) for d in PUBLISHED}


class GateRow:
    """One published-vs-observed comparison."""

    def __init__(self, key: str, expected: float, observed: float | None,
                 tolerance: float, counts: bool) -> None:
        self.key = key
        self.expected = expected
        self.observed = observed
        self.tolerance = tolerance
        self.counts = counts

    @property
    def delta(self) -> float | None:
        return None if self.observed is None else self.observed - self.expected

    @property
    def matched(self) -> bool:
        return self.observed is not None and abs(self.observed - self.expected) < self.tolerance

    def status(self) -> str:
        if self.observed is None:
            return "MISSING"
        if not self.matched:
            return "MISMATCH"
        return "match" if self.counts else "match (not counted)"


def compare(results: dict, domain: str = "iota",
            tolerance: float = DEFAULT_TOLERANCE) -> list[GateRow]:
    """Compare an ``official_baseline.py`` per-domain results.json against PUBLISHED.

    Args:
        results: parsed results.json, e.g. ``{"1a_train": {"NDCG@10": 0.0199, ...}, ...}``.
        domain: which published row to compare against.
        tolerance: strict absolute bound; 5e-5 means the observed value rounds to the
            published 4-dp figure, which is what CLAUDE.md §9's "to 4 decimals" asks for.
    """
    if domain not in PUBLISHED:
        raise KeyError(f"no published figures for domain '{domain}'; "
                       f"known: {', '.join(sorted(PUBLISHED))}")

    counting = set(LOAD_BEARING[domain])
    rows = []
    for key, expected in sorted(PUBLISHED[domain].items()):
        entry = results.get(key)
        observed = entry.get(NDCG_KEY) if isinstance(entry, dict) else None
        rows.append(GateRow(key, expected, observed, tolerance, key in counting))
    return rows


def verdict(rows: list[GateRow]) -> bool:
    """The gate passes when every load-bearing row matches. Zero-valued rows abstain."""
    counting = [r for r in rows if r.counts]
    return bool(counting) and all(r.matched for r in counting)


def render(rows: list[GateRow], domain: str, results: dict) -> str:
    lines = [
        f"Phase 1 gate — domain '{domain}' vs BASELINE_RESULTS.md @ 23093c3",
        f"tolerance ±{rows[0].tolerance:g} (match to 4 decimals)",
        "",
        f"  {'key':<10} {'published':>10} {'observed':>10} {'delta':>10}  {'topics':>6}  status",
        f"  {'-' * 10} {'-' * 10:>10} {'-' * 10:>10} {'-' * 10:>10}  {'-' * 6:>6}  ------",
    ]
    for row in rows:
        entry = results.get(row.key)
        topics = entry.get("num_topics", "?") if isinstance(entry, dict) else "-"
        observed = "     -    " if row.observed is None else f"{row.observed:10.4f}"
        delta = "     -    " if row.delta is None else f"{row.delta:+10.4f}"
        lines.append(
            f"  {row.key:<10} {row.expected:10.4f} {observed} {delta}  {str(topics):>6}  {row.status()}"
        )

    abstained = [r.key for r in rows if not r.counts]
    if abstained:
        lines += [
            "",
            f"  Not counted: {', '.join(abstained)} — published as 0.0000, so a match proves",
            "  nothing (a broken pipeline returns 0.0000 too). Reported for information only.",
        ]
    lines += [
        "",
        "  Scale note: IOTA is ~10 queries split 70/30, so dev is roughly 3 topics.",
        "  These numbers are wiring evidence, not performance evidence.",
        "",
        f"RESULT: {'PASS' if verdict(rows) else 'FAIL'}",
    ]
    return "\n".join(lines)


def print_expected() -> None:
    print("BASELINE_RESULTS.md @ 23093c3 — published per-domain nDCG@10 (organizers' figures)\n")
    print(f"  {'domain':<12} {'1a train':>9} {'1a dev':>9} {'1b train':>9} {'1b dev':>9}")
    print(f"  {'-' * 12} {'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9}")
    for domain, vals in sorted(PUBLISHED.items()):
        print(f"  {domain:<12} {vals['1a_train']:9.4f} {vals['1a_dev']:9.4f} "
              f"{vals['1b_train']:9.4f} {vals['1b_dev']:9.4f}")
    print(f"  {'-' * 12} {'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9} {'-' * 9:>9}")
    print(f"  {'MACRO (13)':<12} {PUBLISHED_MACRO['1a_train']:9.4f} "
          f"{PUBLISHED_MACRO['1a_dev']:9.4f} {PUBLISHED_MACRO['1b_train']:9.4f} "
          f"{PUBLISHED_MACRO['1b_dev']:9.4f}")
    print("\n  Macro is the equal-weight mean over the 13 domains, not over topics.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None,
                        help="results.json from official_baseline.py "
                             "(default: cache/kernel_output/reteco-phase1-bm25-iota/results.json)")
    parser.add_argument("--domain", default="iota", help="published row to compare against")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--expected", action="store_true",
                        help="print the published table and exit")
    args = parser.parse_args()

    if args.expected:
        print_expected()
        return 0

    path = args.results or (cache_root() / "kernel_output" / "reteco-phase1-bm25-iota"
                            / "results.json")
    if not path.is_file():
        print(f"no results at {path}\n\n"
              f"Run the Phase 1 kernel first:\n"
              f"  python kaggle/push_kernel.py kaggle/kernels/phase1_bm25_iota\n"
              f"  python kaggle/pull_output.py --kernel reteco-phase1-bm25-iota",
              file=sys.stderr)
        return 1

    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    rows = compare(results, args.domain, args.tolerance)
    print(render(rows, args.domain, results))
    return 0 if verdict(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
