"""The ledger is the record of record (CLAUDE.md §5.6, §10) — its header must not drift."""

from __future__ import annotations

import csv

from reteco.paths import ledger_path

EXPECTED_COLUMNS = [
    "date",
    "run_id",
    "git_commit",
    "phase",
    "split",
    "subtrack",
    "config",
    "models",
    "macro_ndcg10",
    "ci_low",
    "ci_high",
    "recall100",
    "gpu_hours",
    "hardware",
    "notes",
    "dev_eval_reason",
]


def read_rows() -> list[list[str]]:
    with ledger_path().open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_ledger_exists() -> None:
    assert ledger_path().is_file(), "results/ledger.csv is read at the start of every session (§0)"


def test_header_matches_the_documented_columns() -> None:
    assert read_rows()[0] == EXPECTED_COLUMNS


def test_every_row_has_the_full_column_count() -> None:
    rows = read_rows()
    for number, row in enumerate(rows[1:], start=2):
        assert len(row) == len(EXPECTED_COLUMNS), f"ledger row {number} has {len(row)} fields"


def test_no_result_is_recorded_before_phase_1_passes() -> None:
    """Phase 0 has produced no measurement. A number here before Phase 1's gate is fabricated (§5.6)."""
    assert len(read_rows()) == 1, (
        "the ledger has gained rows — if that was a real logged run, update this test "
        "and PROGRESS.md together"
    )
