#!/usr/bin/env python3
"""Download a finished kernel's output into the repo's gitignored cache.

Usage::

    python kaggle/pull_output.py --kernel rafiurrahman01/reteco-smoke-gpu
    python kaggle/pull_output.py --kernel reteco-smoke-gpu --dest cache/smoke

Output lands in ``cache/kernel_output/<slug>/`` by default (``cache/`` is gitignored per
CLAUDE.md §10), so results come back into the working tree instead of a browser download.
The kernel's console log is printed at the end, which is the whole point for the smoke test.

Runs locally only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cli import KaggleCliError, run_kaggle, setup_logging  # noqa: E402
from creds import CredentialsError, export_to_env  # noqa: E402
from reteco.paths import cache_root, ensure_dir  # noqa: E402

LOG = logging.getLogger("reteco.kaggle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--kernel",
        required=True,
        help="kernel ref, '<username>/<slug>' or just '<slug>' (username filled in from credentials)",
    )
    parser.add_argument("--dest", type=Path, default=None, help="destination directory (default cache/kernel_output/<slug>)")
    parser.add_argument("--log", action="store_true", default=True, help="print the kernel's console log (default on)")
    parser.add_argument("--no-log", dest="log", action="store_false")
    parser.add_argument("--log-file", type=Path, default=None, help="also write this script's logs here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)

    try:
        creds = export_to_env()
    except CredentialsError as exc:
        LOG.error("%s", exc)
        return 1

    kernel_ref = args.kernel if "/" in args.kernel else f"{creds.username}/{args.kernel}"
    slug = kernel_ref.split("/", 1)[1]
    dest = args.dest.resolve() if args.dest else cache_root() / "kernel_output" / slug
    ensure_dir(dest)

    try:
        run_kaggle(["kernels", "output", kernel_ref, "-p", str(dest)])
    except KaggleCliError as exc:
        LOG.error("%s", exc)
        return 1

    files = sorted(p for p in dest.rglob("*") if p.is_file())
    LOG.info("pulled %d file(s) into %s", len(files), dest)
    for path in files:
        LOG.info("  %s (%s bytes)", path.relative_to(dest), path.stat().st_size)

    if args.log:
        logs = [p for p in files if p.suffix == ".log" or p.name.endswith(".log.json")]
        for path in logs:
            LOG.info("----- %s -----", path.name)
            print(path.read_text(encoding="utf-8", errors="replace"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
