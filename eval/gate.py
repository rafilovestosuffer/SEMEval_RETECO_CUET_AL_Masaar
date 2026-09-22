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

__all__ = ["PUBLISHED", "PUBLISHED_MACRO", "LOAD_BEARING", "GateRow", "compare", "verdict",
           "load_results_tree", "macro_from_results", "compare_macro", "render_all"]

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


SUBTRACK_KEYS = ("1a_train", "1a_dev", "1b_train", "1b_dev")


def load_results_tree(root: Path, track: str = "track1_tempo") -> dict[str, dict]:
    """Collect every ``<root>/<track>/<domain>/results.json`` into ``{domain: results}``.

    ``root`` is the ``--out`` directory of `official_baseline.py`, as pulled back from a
    kernel. Domains missing a results.json are simply absent, and `compare_macro` then
    reports a domain count below 13 rather than averaging a hole as zero.
    """
    base = Path(root) / track
    if not base.is_dir():
        base = Path(root)
    out = {}
    for path in sorted(base.glob("*/results.json")):
        out[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def macro_from_results(results: dict[str, dict], key: str) -> tuple[float | None, int]:
    """Equal-weight mean of per-domain nDCG@10 for one sub-track/split.

    **This is the official aggregation** (CLAUDE.md §4): every domain counts once,
    regardless of topic count. A mean over pooled topics is a different number and is
    not what RETECO reports.
    """
    values = [(results[d].get(key) or {}).get(NDCG_KEY) for d in sorted(results)]
    values = [v for v in values if v is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def compare_macro(results: dict[str, dict],
                  tolerance: float = DEFAULT_TOLERANCE) -> list[GateRow]:
    """Compare the macro over domains against the published macro table.

    A precision note, because it looks like a bug the first time it bites. The published
    *macro* is a 4-dp rounding of the organizers' full-precision mean. Our macro is a mean
    of the per-domain values in ``results.json``, which `official_baseline.py` writes to 5
    dp — so our rounding error is at most ~5e-6 and the 5e-5 bound is comfortable.

    Averaging the 4-dp figures from BASELINE_RESULTS.md instead is *not* equivalent: those
    13 values average to 0.087846 for 1a_train, which rounds to the published 0.0879 but
    sits 5.4e-5 away from it — outside this tolerance. So feed this function real
    ``results.json`` output, never the published per-domain table.
    """
    rows = []
    for key in SUBTRACK_KEYS:
        observed, _ = macro_from_results(results, key)
        rows.append(GateRow(key, PUBLISHED_MACRO[key], observed, tolerance, counts=True))
    return rows


def render_all(results: dict[str, dict], tolerance: float = DEFAULT_TOLERANCE) -> str:
    """Per-domain table plus the macro line — the Phase 2 gate (CLAUDE.md §9)."""
    lines = [
        "Phase 2 gate — all Track 1 domains vs BASELINE_RESULTS.md @ 23093c3",
        f"tolerance ±{tolerance:g} (match to 4 decimals)",
        "",
        f"  {'domain':<12}" + "".join(f"{k:>18}" for k in SUBTRACK_KEYS),
        f"  {'-' * 12}" + "".join(f"{'-' * 18:>18}" for _ in SUBTRACK_KEYS),
    ]
    for domain in sorted(results):
        cells = []
        for key in SUBTRACK_KEYS:
            expected = PUBLISHED.get(domain, {}).get(key)
            observed = (results[domain].get(key) or {}).get(NDCG_KEY)
            counts = expected is not None and expected != 0.0
            ok = (observed is not None and expected is not None
                  and abs(observed - expected) < tolerance)
            mark = "ok" if ok else ("--" if not counts else "XX")
            shown = f"{observed:.4f}" if observed is not None else "  -   "
            cells.append(f"{shown} {mark:>3}")
        lines.append(f"  {domain:<12}" + "".join(f"{c:>18}" for c in cells))

    missing = sorted(set(PUBLISHED) - set(results))
    if missing:
        lines.append(f"\n  MISSING domains (no results.json): {', '.join(missing)}")

    macro_rows = compare_macro(results, tolerance)
    lines += [
        "",
        f"  {'macro':<10} {'published':>10} {'observed':>10} {'delta':>10}  status",
        f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}  ------",
    ]
    for row in macro_rows:
        observed = "     -    " if row.observed is None else f"{row.observed:10.4f}"
        delta = "     -    " if row.delta is None else f"{row.delta:+10.4f}"
        lines.append(f"  {row.key:<10} {row.expected:10.4f} {observed} {delta}  "
                     f"{row.status()}")

    _, n_domains = macro_from_results(results, "1a_train")
    lines += [
        "",
        f"  Macro is the equal-weight mean over {n_domains} domains, not over topics (§4).",
        "  Cells are observed nDCG@10; 'ok' matched to 4 dp, 'XX' diverged, '--' is a",
        "  published 0.0000 which abstains — a broken pipeline reproduces it by accident.",
        "",
        f"RESULT: {'PASS' if verdict(macro_rows) else 'FAIL'}",
    ]
    return "\n".join(lines)


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
    parser.add_argument("--all", action="store_true",
                        help="Phase 2 gate: every domain plus the macro, from --results-dir")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="official_baseline.py --out directory (holds <track>/<domain>/results.json)")
    args = parser.parse_args()

    if args.expected:
        print_expected()
        return 0

    if args.all:
        root = args.results_dir or (cache_root() / "kernel_output"
                                    / "reteco-phase2-bm25-full" / "baseline_out")
        results = load_results_tree(root)
        if not results:
            print(f"no per-domain results.json under {root}\n\n"
                  f"Run the Phase 2 kernel first:\n"
                  f"  python kaggle/push_kernel.py kaggle/kernels/phase2_bm25_full\n"
                  f"  python kaggle/pull_output.py --kernel reteco-phase2-bm25-full",
                  file=sys.stderr)
            return 1
        print(render_all(results, args.tolerance))
        return 0 if verdict(compare_macro(results, args.tolerance)) else 1

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
