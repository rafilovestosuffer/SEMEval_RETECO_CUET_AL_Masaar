"""Path resolution must behave identically on Git Bash and Kaggle (CLAUDE.md §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reteco import paths

ENV_VARS = ("RETECO_DATA", "RETECO_CACHE", "RETECO_RESULTS", "KAGGLE_KERNEL_RUN_TYPE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a known environment, whatever the host has set."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_repo_root_is_independent_of_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    before = paths.repo_root()
    monkeypatch.chdir(tmp_path)
    assert paths.repo_root() == before
    assert (paths.repo_root() / "CLAUDE.md").is_file()


def test_local_defaults_live_under_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "on_kaggle", lambda: False)
    root = paths.repo_root()
    assert paths.data_root() == root / "reteco_data"
    assert paths.cache_root() == root / "cache"
    assert paths.results_root() == root / "results"


def test_kaggle_defaults_split_input_from_working(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Kaggle, data is read-only under /kaggle/input and only /kaggle/working persists."""
    monkeypatch.setattr(paths, "on_kaggle", lambda: True)
    assert paths.data_root() == paths.KAGGLE_INPUT
    assert paths.cache_root() == paths.KAGGLE_WORKING / "cache"
    assert paths.results_root() == paths.KAGGLE_WORKING / "results"


def test_env_override_beats_both_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "on_kaggle", lambda: True)
    monkeypatch.setenv("RETECO_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()


def test_on_kaggle_detects_the_run_type_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert paths.on_kaggle() is True


def test_domain_dir_matches_the_documented_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLAUDE.md §4 gives reteco_data/track1_tempo/iota/qrels_dev.txt as the path shape."""
    monkeypatch.setenv("RETECO_DATA", str(tmp_path))
    assert paths.domain_dir("iota") == tmp_path.resolve() / "track1_tempo" / "iota"
    assert paths.domain_dir("iota").parts[-2:] == ("track1_tempo", "iota")


def test_ledger_stays_in_the_repo_even_on_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ledger is version-controlled; a kernel must not fork it into ephemeral storage."""
    monkeypatch.setattr(paths, "on_kaggle", lambda: True)
    assert paths.ledger_path() == paths.repo_root() / "results" / "ledger.csv"


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    assert paths.ensure_dir(target) == target
    assert paths.ensure_dir(target) == target
    assert target.is_dir()
