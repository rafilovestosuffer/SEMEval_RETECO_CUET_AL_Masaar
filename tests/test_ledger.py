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


def test_every_row_names_the_commit_that_produced_it() -> None:
    """§5.7: a run is only reproducible if the ledger says which code produced it.

    Replaced the pre-Phase-1 "the ledger must be empty" guard on 2026-09-22, when the
    Phase 1 gate passed on IOTA and the first real rows were logged. That guard's job
    was to catch a number invented before any measurement existed; from here on the
    equivalent protection is that every number is traceable to a commit.
    """
    for number, row in enumerate(read_rows()[1:], start=2):
        entry = dict(zip(EXPECTED_COLUMNS, row))
        assert entry["git_commit"].strip(), f"ledger row {number} has no git_commit"


def test_every_row_carries_a_measurement_of_some_kind() -> None:
    """A row must record either a score or a compute cost — never neither.

    Relaxed on 2026-09-22 from "every row has a macro_ndcg10". The Phase 4a throughput
    benchmark is a legitimate run to log under §5.7 (gpu_hours, hardware) but ran no
    retrieval, so it has no nDCG. Forcing a number into that column would have meant
    inventing one, which is precisely what §5.6 forbids. The protection that matters is
    that a row cannot be empty of evidence altogether.
    """
    for number, row in enumerate(read_rows()[1:], start=2):
        entry = dict(zip(EXPECTED_COLUMNS, row))
        has_score = bool(entry["macro_ndcg10"].strip())
        has_cost = bool(entry["gpu_hours"].strip())
        assert has_score or has_cost, (
            f"ledger row {number} records neither a score nor a compute cost"
        )


def test_every_dev_row_states_why_dev_was_touched() -> None:
    """§5.2: dev is sacred — every dev evaluation is logged *with a reason*."""
    for number, row in enumerate(read_rows()[1:], start=2):
        entry = dict(zip(EXPECTED_COLUMNS, row))
        if entry["split"].strip() == "dev":
            assert entry["dev_eval_reason"].strip(), (
                f"ledger row {number} evaluates on dev without a stated reason"
            )
