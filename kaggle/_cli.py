"""Thin, logged wrapper around the ``kaggle`` command-line tool.

Everything here shells out to the ``kaggle`` console script rather than importing the
``kaggle`` Python package. That is deliberate: this directory is named ``kaggle/``
(CLAUDE.md §10), so from the repo root ``import kaggle`` resolves to *this* directory
as a namespace package instead of the installed library. Subprocess calls go through
PATH and are immune to that shadowing.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["KaggleCliError", "require_cli", "run_kaggle", "setup_logging"]

LOG = logging.getLogger("reteco.kaggle")

CLI_MISSING_HELP = """\
The 'kaggle' command was not found on PATH.

  pip install -r requirements.txt

Note: run that from OUTSIDE this repo's root, or use `python -m pip`, and do not test the
install with `python -c "import kaggle"` while your shell is in the repo root — the local
kaggle/ directory shadows the installed package. `kaggle --version` is the reliable check.
"""


class KaggleCliError(RuntimeError):
    """Raised when a kaggle CLI invocation fails."""


def setup_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    """Configure logging to stderr and, optionally, to a file (CLAUDE.md §11)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def require_cli() -> str:
    """Return the path to the ``kaggle`` executable, or raise with install instructions."""
    exe = shutil.which("kaggle")
    if exe is None:
        raise KaggleCliError(CLI_MISSING_HELP)
    return exe


def run_kaggle(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``kaggle <args>`` and return the completed process.

    stdout and stderr are captured and logged rather than streamed, so the caller can
    parse them (kernel status, dataset existence) and so a failed call leaves a readable
    record in the log file.

    Args:
        args: arguments after the ``kaggle`` executable, e.g. ``["kernels", "push", "-p", "."]``.
        check: raise ``KaggleCliError`` on a non-zero exit code.
    """
    exe = require_cli()
    LOG.debug("running: kaggle %s", " ".join(args))
    proc = subprocess.run(  # noqa: S603 - fixed executable, arguments are not shell-interpreted
        [exe, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in (proc.stdout or "").splitlines():
        LOG.debug("  out: %s", line)
    for line in (proc.stderr or "").splitlines():
        LOG.debug("  err: %s", line)
    if check and proc.returncode != 0:
        raise KaggleCliError(
            f"`kaggle {' '.join(args)}` exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc
