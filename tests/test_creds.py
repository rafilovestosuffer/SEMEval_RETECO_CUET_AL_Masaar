"""Credential resolution, and the guarantee that no key ever reaches a log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import creds

SECRET = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    # Point at an empty dir so a real ~/.kaggle/kaggle.json on the host cannot leak in.
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "empty"))


def write_kaggle_json(directory: Path, username: str = "rafiurrahman01", key: str = SECRET) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kaggle.json"
    path.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_env_vars_win_over_kaggle_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_kaggle_json(tmp_path / "cfg", username="from_file", key="file_key")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("KAGGLE_USERNAME", "from_env")
    monkeypatch.setenv("KAGGLE_KEY", SECRET)

    resolved = creds.load_credentials()
    assert resolved.username == "from_env"
    assert resolved.source == "environment"


def test_falls_back_to_kaggle_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = write_kaggle_json(tmp_path / "cfg")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))

    resolved = creds.load_credentials()
    assert resolved.username == "rafiurrahman01"
    assert resolved.key == SECRET
    assert resolved.source == str(path)


def test_partial_env_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Username alone must not be treated as configured — it would fail confusingly later."""
    monkeypatch.setenv("KAGGLE_USERNAME", "rafiurrahman01")
    with pytest.raises(creds.CredentialsError):
        creds.load_credentials()


def test_missing_credentials_explain_the_fix() -> None:
    with pytest.raises(creds.CredentialsError) as excinfo:
        creds.load_credentials()
    message = str(excinfo.value)
    assert "kaggle.com/settings" in message
    assert "chmod 600" in message


def test_malformed_json_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "kaggle.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg))
    with pytest.raises(creds.CredentialsError, match="could not be read as JSON"):
        creds.load_credentials()


def test_json_missing_key_field_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "kaggle.json").write_text(json.dumps({"username": "rafi"}), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg))
    with pytest.raises(creds.CredentialsError, match="missing"):
        creds.load_credentials()


def test_describe_never_contains_the_key() -> None:
    """A pasted terminal log must not leak the secret — this is the whole point of describe()."""
    resolved = creds.KaggleCreds(username="rafiurrahman01", key=SECRET, source="test")
    described = resolved.describe()
    assert SECRET not in described
    assert "rafiurrahman01" in described
    assert described.endswith("(len=32)")
    # Only the last two characters survive.
    assert SECRET[-2:] in described
    assert SECRET[:8] not in described


def test_export_to_env_populates_the_cli_variables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_kaggle_json(tmp_path / "cfg")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "cfg"))

    resolved = creds.export_to_env()
    import os

    assert os.environ["KAGGLE_USERNAME"] == resolved.username
    assert os.environ["KAGGLE_KEY"] == SECRET


def test_config_dir_honours_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert creds.config_dir() == tmp_path
