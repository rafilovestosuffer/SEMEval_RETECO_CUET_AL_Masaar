"""Path resolution that works identically on Git Bash (Windows) and Kaggle Linux.

CLAUDE.md §6 requires every script to run on both machines with no hard-coded
separators, so all path construction goes through this module rather than being
re-derived per script.

Three roots, each overridable by an environment variable:

============  ==================  ===========================  ==========================
root          env var             local default                Kaggle default
============  ==================  ===========================  ==========================
``data``      ``RETECO_DATA``     ``<repo>/reteco_data``       ``/kaggle/input``
``cache``     ``RETECO_CACHE``    ``<repo>/cache``             ``/kaggle/working/cache``
``results``   ``RETECO_RESULTS``  ``<repo>/results``           ``/kaggle/working/results``
============  ==================  ===========================  ==========================

The Kaggle defaults differ for a reason: ``/kaggle/input`` is read-only and holds
attached datasets, while only ``/kaggle/working`` persists into the kernel's output.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "KAGGLE_INPUT",
    "KAGGLE_WORKING",
    "on_kaggle",
    "repo_root",
    "data_root",
    "cache_root",
    "results_root",
    "runs_dir",
    "ledger_path",
    "domain_dir",
    "ensure_dir",
]

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Track 1 domain directories live under this prefix inside the dataset, e.g.
# reteco_data/track1_tempo/iota/qrels_dev.txt  (CLAUDE.md §4).
TRACK1_PREFIX = "track1_tempo"


def on_kaggle() -> bool:
    """True when running inside a Kaggle kernel.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE``; the directory check is a fallback for
    images where the variable is absent.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return KAGGLE_WORKING.is_dir() and KAGGLE_INPUT.is_dir()


def repo_root() -> Path:
    """Absolute path of the repository root.

    Resolved from this file's location (``<repo>/reteco/paths.py``) so it is correct
    regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


def _root_from_env(var: str, kaggle_default: Path, local_default: Path) -> Path:
    override = os.environ.get(var)
    if override:
        return Path(override).expanduser().resolve()
    return kaggle_default if on_kaggle() else local_default.resolve()


def data_root() -> Path:
    """Where the organizer corpus lives. Read-only on Kaggle."""
    return _root_from_env("RETECO_DATA", KAGGLE_INPUT, repo_root() / "reteco_data")


def cache_root() -> Path:
    """Embeddings, LLM generations, pulled kernel output. Gitignored; never a source of truth."""
    return _root_from_env("RETECO_CACHE", KAGGLE_WORKING / "cache", repo_root() / "cache")


def results_root() -> Path:
    """TREC runs and the ledger."""
    return _root_from_env("RETECO_RESULTS", KAGGLE_WORKING / "results", repo_root() / "results")


def runs_dir() -> Path:
    """Directory holding TREC run files."""
    return results_root() / "runs"


def ledger_path() -> Path:
    """The experiment ledger (CLAUDE.md §10).

    Always the repo copy, even on Kaggle: the ledger is version-controlled and must not
    be forked into a kernel's ephemeral working directory.
    """
    return repo_root() / "results" / "ledger.csv"


def domain_dir(domain: str, track: str = TRACK1_PREFIX) -> Path:
    """Directory for one Track 1 domain, e.g. ``domain_dir("iota")``.

    Args:
        domain: domain name as it appears in the dataset, e.g. ``"iota"``, ``"history"``.
        track: track subdirectory; defaults to Track 1 (TEMPO).
    """
    return data_root() / track / domain


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing, then return it. Idempotent (§11)."""
    path.mkdir(parents=True, exist_ok=True)
    return path
