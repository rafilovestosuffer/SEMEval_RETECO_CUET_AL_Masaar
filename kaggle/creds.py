"""Locate Kaggle API credentials without ever storing them in this repository.

Resolution order (first hit wins):

1. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables.
2. ``~/.kaggle/kaggle.json`` (or ``$KAGGLE_CONFIG_DIR/kaggle.json``).

The repository tree is deliberately *not* searched: a key that lives in the repo is a
key that eventually gets committed. ``.gitignore`` blocks ``kaggle.json`` as a second
line of defence.

Nothing here prints, logs, or returns the key itself — only the username and a
redacted fingerprint, so a traceback or a pasted terminal log never leaks the secret.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["KaggleCreds", "CredentialsError", "config_dir", "load_credentials", "export_to_env"]

SETUP_HELP = """\
Kaggle credentials not found.

Set them up once on this machine:

  1. https://www.kaggle.com/settings  ->  Account  ->  API  ->  "Create New Token"
     (this downloads kaggle.json)
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

Or export them for the current shell instead:

  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_key

Never put the key in this repository, in a kernel script, or in kernel-metadata.json.
"""


class CredentialsError(RuntimeError):
    """Raised when Kaggle credentials are missing or malformed."""


@dataclass(frozen=True)
class KaggleCreds:
    """Resolved Kaggle credentials plus where they came from."""

    username: str
    key: str
    source: str

    def fingerprint(self) -> str:
        """A redacted stand-in for the key, safe to print in logs.

        Shows length and the last two characters only — enough to tell two keys apart
        when debugging a rotation, useless to anyone who reads it.
        """
        if len(self.key) <= 2:
            return "*" * len(self.key)
        return f"{'*' * (len(self.key) - 2)}{self.key[-2:]} (len={len(self.key)})"

    def describe(self) -> str:
        """One-line, leak-safe summary."""
        return f"Kaggle user '{self.username}' from {self.source}; key {self.fingerprint()}"


def config_dir() -> Path:
    """Directory holding ``kaggle.json``, honouring ``KAGGLE_CONFIG_DIR``."""
    override = os.environ.get("KAGGLE_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".kaggle"


def _warn_if_world_readable(path: Path) -> None:
    """Warn when kaggle.json is readable by anyone but its owner.

    Skipped on Windows, where POSIX permission bits are not meaningful.
    """
    if sys.platform.startswith("win"):
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"warning: {path} is readable beyond its owner (mode {oct(stat.S_IMODE(mode))}). "
            f"Run: chmod 600 {path}",
            file=sys.stderr,
        )


def load_credentials() -> KaggleCreds:
    """Return Kaggle credentials, or raise ``CredentialsError`` with setup instructions."""
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return KaggleCreds(username=env_user, key=env_key, source="environment")

    json_path = config_dir() / "kaggle.json"
    if not json_path.is_file():
        raise CredentialsError(SETUP_HELP)

    _warn_if_world_readable(json_path)

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialsError(f"{json_path} could not be read as JSON: {exc}\n\n{SETUP_HELP}") from exc

    username = payload.get("username")
    key = payload.get("key")
    if not username or not key:
        raise CredentialsError(
            f"{json_path} is missing 'username' and/or 'key'.\n\n{SETUP_HELP}"
        )

    return KaggleCreds(username=str(username), key=str(key), source=str(json_path))


def export_to_env() -> KaggleCreds:
    """Load credentials and place them in ``os.environ`` for the Kaggle CLI.

    The CLI reads ``KAGGLE_USERNAME``/``KAGGLE_KEY`` before ``kaggle.json``, so doing
    this up front makes every subprocess call in these scripts use the same identity
    that was resolved and reported here.
    """
    creds = load_credentials()
    os.environ["KAGGLE_USERNAME"] = creds.username
    os.environ["KAGGLE_KEY"] = creds.key
    return creds


if __name__ == "__main__":
    try:
        print(load_credentials().describe())
    except CredentialsError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
