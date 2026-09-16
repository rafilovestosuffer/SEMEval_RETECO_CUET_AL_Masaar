#!/usr/bin/env python3
"""Create or version a PRIVATE Kaggle Dataset from a local directory.

Usage::

    # first upload — creates the dataset
    python kaggle/upload_dataset.py --dir reteco_data --slug reteco-track1 \
        --title "RETECO Track 1 (TEMPO)" --dry-run
    python kaggle/upload_dataset.py --dir reteco_data --slug reteco-track1 \
        --title "RETECO Track 1 (TEMPO)"

    # later uploads — adds a new version
    python kaggle/upload_dataset.py --dir cache/embeddings --slug reteco-cache \
        --title "RETECO embedding cache" --message "bge-m3, domains 1-6"

Why datasets at all (CLAUDE.md §6): a Kaggle kernel is a fresh container every run, and
re-downloading 1.65M documents inside each one burns wall clock against the 12h session
cap. Uploading once and attaching the dataset makes the corpus available instantly, and
versioning an *output* dataset back up is how a job resumes after a kill — the next run
attaches it and skips the domains already finished.

Datasets are created private (``--public`` is required to opt out). The organizer corpus
is CC BY-SA 4.0 / CC BY 4.0 but redistributing it publicly is not our call to make.

Runs locally only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import KaggleCliError, run_kaggle, setup_logging  # noqa: E402
from creds import CredentialsError, export_to_env  # noqa: E402

LOG = logging.getLogger("reteco.kaggle")

METADATA_NAME = "dataset-metadata.json"


def human_bytes(n: int) -> str:
    """Format a byte count for logs."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def survey(directory: Path) -> tuple[int, int]:
    """Return (file count, total bytes) for everything under ``directory``."""
    count = 0
    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and path.name != METADATA_NAME:
            count += 1
            total += path.stat().st_size
    return count, total


def write_metadata(directory: Path, owner: str, slug: str, title: str) -> Path:
    """Write ``dataset-metadata.json`` into ``directory`` (the CLI reads it from there).

    Idempotent: rewriting it with the same inputs produces the same file.
    """
    meta_path = directory / METADATA_NAME
    meta = {"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "unknown"}]}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def dataset_exists(ref: str) -> bool:
    """True when a dataset with this ref is already visible to the account."""
    proc = run_kaggle(["datasets", "status", ref], check=False)
    if proc.returncode != 0:
        return False
    combined = f"{proc.stdout}{proc.stderr}".lower()
    return "404" not in combined and "not found" not in combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", dest="directory", type=Path, required=True, help="local directory to upload")
    parser.add_argument("--slug", required=True, help="dataset slug, e.g. reteco-track1")
    parser.add_argument("--title", required=True, help="human-readable dataset title")
    parser.add_argument("--message", default="update", help="version message (ignored on first create)")
    parser.add_argument("--public", action="store_true", help="make the dataset public (default: private)")
    parser.add_argument("--dir-mode", default="zip", choices=("skip", "zip", "tar"), help="how to handle subdirectories (default zip)")
    parser.add_argument("--dry-run", action="store_true", help="write metadata and report size, upload nothing")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)

    directory = args.directory.resolve()
    if not directory.is_dir():
        LOG.error("not a directory: %s", directory)
        return 1

    count, total = survey(directory)
    if count == 0:
        LOG.error("%s contains no files to upload", directory)
        return 1
    LOG.info("%s: %d file(s), %s", directory, count, human_bytes(total))

    try:
        creds = export_to_env()
    except CredentialsError as exc:
        LOG.error("%s", exc)
        return 1

    ref = f"{creds.username}/{args.slug}"
    meta_path = write_metadata(directory, creds.username, args.slug, args.title)
    LOG.info("wrote %s", meta_path)

    if args.dry_run:
        LOG.info("--dry-run: would upload as %s (%s)", ref, "public" if args.public else "private")
        LOG.info("metadata:\n%s", meta_path.read_text(encoding="utf-8"))
        return 0

    exists = dataset_exists(ref)
    if exists:
        LOG.info("%s exists -> creating a new version", ref)
        cmd = ["datasets", "version", "-p", str(directory), "-m", args.message, "--dir-mode", args.dir_mode]
    else:
        LOG.info("%s does not exist -> creating it (%s)", ref, "public" if args.public else "private")
        cmd = ["datasets", "create", "-p", str(directory), "--dir-mode", args.dir_mode]
        if not args.public:
            cmd.append("--private")

    try:
        proc = run_kaggle(cmd)
    except KaggleCliError as exc:
        LOG.error("%s", exc)
        return 1

    LOG.info("%s", (proc.stdout or "").strip())
    LOG.info("attach it to a kernel by adding \"%s\" to dataset_sources in kernel-metadata.json", ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
