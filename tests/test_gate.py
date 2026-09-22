"""The Phase 1 gate (CLAUDE.md §9) and the integrity of the published figures it checks against."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import gate
from reteco.paths import repo_root

SUBTRACK_KEYS = ["1a_train", "1a_dev", "1b_train", "1b_dev"]


def results_for(**ndcg: float) -> dict:
    """Build a results.json payload shaped like official_baseline.py's output."""
    return {k: {"NDCG@10": v, "num_topics": 7} for k, v in ndcg.items()}


def iota_exact() -> dict:
    return results_for(**gate.PUBLISHED["iota"])


# --------------------------------------------------------------- published table --
def test_all_thirteen_domains_present() -> None:
    """The macro-average is over 13 domains; a missing one would silently skew Phase 2."""
    assert len(gate.PUBLISHED) == 13


def test_every_domain_has_every_subtrack() -> None:
    for domain, vals in gate.PUBLISHED.items():
        assert sorted(vals) == sorted(SUBTRACK_KEYS), f"{domain} is incomplete"


def test_per_domain_values_average_to_the_published_macro() -> None:
    """Cross-check of the transcription from BASELINE_RESULTS.md.

    Each per-domain figure is published rounded to 4 dp, so the mean of the 13 carries
    up to ~5e-5 of rounding error against the separately-rounded published macro.
    Anything larger is a transcription mistake, not rounding.
    """
    for key in SUBTRACK_KEYS:
        values = [d[key] for d in gate.PUBLISHED.values()]
        mean = sum(values) / len(values)
        assert mean == pytest.approx(gate.PUBLISHED_MACRO[key], abs=1e-4), key


def test_iota_1b_train_is_the_documented_zero() -> None:
    """The abstention logic exists because of this specific value; pin it."""
    assert gate.PUBLISHED["iota"]["1b_train"] == 0.0


# ------------------------------------------------------------------- comparison --
def test_exact_published_values_pass() -> None:
    rows = gate.compare(iota_exact())
    assert gate.verdict(rows) is True
    assert all(r.matched for r in rows)


def test_near_miss_at_the_fourth_decimal_fails() -> None:
    """'Matches to 4 decimals' is the gate (§9) — 0.0200 vs 0.0199 must not pass."""
    payload = iota_exact()
    payload["1a_train"]["NDCG@10"] = 0.0200
    rows = gate.compare(payload)
    assert gate.verdict(rows) is False
    assert next(r for r in rows if r.key == "1a_train").status() == "MISMATCH"


def test_difference_that_still_rounds_to_the_published_value_passes() -> None:
    """0.01993 rounds to 0.0199, so it matches to 4 decimals."""
    payload = iota_exact()
    payload["1a_train"]["NDCG@10"] = 0.0199 + 3e-5
    assert gate.verdict(gate.compare(payload)) is True


def test_tolerance_is_a_half_ulp_not_a_full_one() -> None:
    """Guards the bug this suite caught: 1e-4 with <= would let 0.0200 match 0.0199."""
    assert gate.DEFAULT_TOLERANCE == 5e-5
    payload = iota_exact()
    payload["1a_train"]["NDCG@10"] = 0.0199 + 6e-5  # 0.01996 -> rounds to 0.0200
    assert gate.verdict(gate.compare(payload)) is False


def test_missing_key_is_reported_and_fails() -> None:
    payload = iota_exact()
    del payload["1a_dev"]
    rows = gate.compare(payload)
    row = next(r for r in rows if r.key == "1a_dev")
    assert row.observed is None
    assert row.status() == "MISSING"
    assert row.delta is None
    assert gate.verdict(rows) is False


def test_zero_valued_row_abstains_from_the_verdict() -> None:
    """A published 0.0000 must neither pass nor fail the gate on its own."""
    assert "1b_train" not in gate.LOAD_BEARING["iota"]
    assert sorted(gate.LOAD_BEARING["iota"]) == ["1a_dev", "1a_train", "1b_dev"]


def test_an_all_zero_run_fails_despite_matching_the_zero_row() -> None:
    """The exact failure the abstention rule exists to catch: a broken pipeline."""
    payload = results_for(**{k: 0.0 for k in SUBTRACK_KEYS})
    rows = gate.compare(payload)
    assert next(r for r in rows if r.key == "1b_train").matched is True
    assert gate.verdict(rows) is False


def test_empty_results_fail_rather_than_vacuously_pass() -> None:
    assert gate.verdict(gate.compare({})) is False


def test_unknown_domain_raises() -> None:
    with pytest.raises(KeyError, match="no published figures"):
        gate.compare(iota_exact(), domain="atlantis")


def test_non_dict_entry_is_treated_as_missing() -> None:
    payload = iota_exact()
    payload["1a_train"] = None
    assert next(r for r in gate.compare(payload) if r.key == "1a_train").observed is None


# ----------------------------------------------------------------------- render --
def test_render_marks_the_abstaining_row_and_the_verdict() -> None:
    text = gate.render(gate.compare(iota_exact()), "iota", iota_exact())
    assert "RESULT: PASS" in text
    assert "not counted" in text.lower()
    assert "1b_train" in text


# -------------------------------------------------------------------------- cli --
def test_cli_expected_lists_all_domains_and_the_macro() -> None:
    proc = subprocess.run(
        [sys.executable, str(repo_root() / "eval" / "gate.py"), "--expected"],
        capture_output=True, text=True, check=True,
    )
    for domain in gate.PUBLISHED:
        assert domain in proc.stdout
    assert "MACRO (13)" in proc.stdout


def test_cli_exit_codes_follow_the_verdict(tmp_path: Path) -> None:
    good = tmp_path / "pass.json"
    good.write_text(json.dumps(iota_exact()), encoding="utf-8")
    bad_payload = iota_exact()
    bad_payload["1a_dev"]["NDCG@10"] = 0.9
    bad = tmp_path / "fail.json"
    bad.write_text(json.dumps(bad_payload), encoding="utf-8")

    script = str(repo_root() / "eval" / "gate.py")
    assert subprocess.run([sys.executable, script, "--results", str(good)],
                          capture_output=True, text=True).returncode == 0
    assert subprocess.run([sys.executable, script, "--results", str(bad)],
                          capture_output=True, text=True).returncode == 1


def test_cli_missing_results_explains_how_to_produce_them(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(repo_root() / "eval" / "gate.py"),
         "--results", str(tmp_path / "nope.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "push_kernel.py" in proc.stderr


# ---------------------------------------------------------------- Phase 2 gate --
def _tree(tmp_path: Path, per_domain: dict[str, dict[str, float]]) -> Path:
    """Write a fake official_baseline.py --out tree: <root>/track1_tempo/<dom>/results.json."""
    root = tmp_path / "baseline_out"
    for domain, values in per_domain.items():
        directory = root / "track1_tempo" / domain
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(
            json.dumps({k: {"NDCG@10": v, "num_topics": 5} for k, v in values.items()}),
            encoding="utf-8",
        )
    return root


def test_macro_is_equal_weight_over_domains_not_topics(tmp_path: Path) -> None:
    """§4: a 1000-topic domain and a 3-topic domain each count exactly once."""
    root = tmp_path / "baseline_out"
    for domain, score, topics in (("history", 0.0, 1000), ("iota", 1.0, 3)):
        directory = root / "track1_tempo" / domain
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(
            json.dumps({"1a_train": {"NDCG@10": score, "num_topics": topics}}),
            encoding="utf-8",
        )
    results = gate.load_results_tree(root)
    macro, n = gate.macro_from_results(results, "1a_train")
    assert n == 2
    assert macro == pytest.approx(0.5), "topic counts must not weight the macro"


def test_all_domains_reporting_published_figures_passes(tmp_path: Path) -> None:
    root = _tree(tmp_path, gate.PUBLISHED)
    results = gate.load_results_tree(root)
    assert len(results) == 13
    for domain in gate.PUBLISHED:
        assert gate.verdict(gate.compare(results[domain], domain))


def test_a_single_diverging_domain_fails_the_macro(tmp_path: Path) -> None:
    perturbed = {d: dict(v) for d, v in gate.PUBLISHED.items()}
    perturbed["history"]["1a_dev"] = 0.9
    results = gate.load_results_tree(_tree(tmp_path, perturbed))
    rows = {r.key: r for r in gate.compare_macro(results)}
    assert not rows["1a_dev"].matched


def test_missing_domain_is_reported_not_averaged_as_zero(tmp_path: Path) -> None:
    subset = {d: v for d, v in gate.PUBLISHED.items() if d != "history"}
    results = gate.load_results_tree(_tree(tmp_path, subset))
    _, n = gate.macro_from_results(results, "1a_train")
    assert n == 12, "a missing domain must shrink the denominator, not count as 0.0"
    assert "MISSING" in gate.render_all(results)
    assert "history" in gate.render_all(results)
