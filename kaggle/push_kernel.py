#!/usr/bin/env python3
"""Push a kernel directory to Kaggle and poll until it finishes.

Usage::

    python kaggle/push_kernel.py kaggle/kernels/smoke_gpu
    python kaggle/push_kernel.py kaggle/kernels/smoke_gpu --no-wait
    python kaggle/push_kernel.py kaggle/kernels/smoke_gpu --dry-run

A kernel directory holds ``kernel-metadata.json`` plus the script it names in
``code_file``. The metadata's ``id`` must be ``<username>/<slug>``; if the username
placeholder still reads ``USERNAME`` it is filled in from the resolved credentials, so
the committed metadata never carries an account name it does not need to.

Runs locally only — a Claude Code web container cannot reach kaggle.com.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import KaggleCliError, run_kaggle, setup_logging  # noqa: E402
from creds import CredentialsError, export_to_env  # noqa: E402

import logging  # noqa: E402

LOG = logging.getLogger("reteco.kaggle")

# Kaggle reports one of these in `kernels status`. Anything else means still running.
TERMINAL_STATES = {"complete", "error", "cancelAcknowledged", "cancelRequested"}
USERNAME_PLACEHOLDER = "USERNAME"

# `kernels push` prints the canonical URL it published to, e.g.
#   Please check progress at https://www.kaggle.com/code/someone/my-kernel-slug
PUSHED_URL = re.compile(r"kaggle\.com/code/([\w-]+/[\w-]+)")


def published_ref(push_output: str, fallback: str) -> str:
    """The kernel ref Kaggle actually published to, per its own output.

    Kaggle derives the slug from the kernel *title*, not from the ``id`` in our metadata,
    and only warns when the two disagree. Polling the declared id then fails with
    "Permission 'kernels.get' was denied" -- which looks like an auth problem and is
    really a wrong slug -- and the push sits there until the timeout expires. Trusting
    the printed URL over our own metadata makes the poll correct even when they diverge.
    """
    match = PUSHED_URL.search(push_output or "")
    return match.group(1) if match else fallback


def load_metadata(kernel_dir: Path) -> dict:
    """Read and sanity-check ``kernel-metadata.json``."""
    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no kernel-metadata.json in {kernel_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    for field in ("id", "title", "code_file", "language", "kernel_type"):
        if field not in meta:
            raise ValueError(f"{meta_path} is missing required field '{field}'")

    code_file = kernel_dir / meta["code_file"]
    if not code_file.is_file():
        raise FileNotFoundError(f"{meta_path} names code_file '{meta['code_file']}', which does not exist")

    return meta


def resolve_id(meta: dict, username: str, kernel_dir: Path) -> str:
    """Return the kernel's full ``<username>/<slug>`` id, substituting the placeholder.

    When ``id`` starts with the placeholder the real username is written back into the
    metadata file, because the Kaggle CLI reads the file itself rather than our copy.
    """
    kernel_id = str(meta["id"])
    if not kernel_id.startswith(f"{USERNAME_PLACEHOLDER}/"):
        return kernel_id

    kernel_id = kernel_id.replace(f"{USERNAME_PLACEHOLDER}/", f"{username}/", 1)
    meta["id"] = kernel_id
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info("filled in username placeholder -> %s", kernel_id)
    return kernel_id


def poll_status(kernel_id: str, interval: int, timeout: int) -> str:
    """Poll ``kaggle kernels status`` until terminal or timeout. Returns the last status text."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        proc = run_kaggle(["kernels", "status", kernel_id], check=False)
        last = (proc.stdout or proc.stderr or "").strip()
        lowered = last.lower()
        if any(state.lower() in lowered for state in TERMINAL_STATES):
            LOG.info("status: %s", last)
            return last
        LOG.info("status: %s  (polling every %ds)", last or "unknown", interval)
        time.sleep(interval)
    LOG.warning("timed out after %ds waiting for %s", timeout, kernel_id)
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel_dir", type=Path, help="directory containing kernel-metadata.json")
    parser.add_argument("--no-wait", action="store_true", help="push and exit without polling")
    parser.add_argument("--dry-run", action="store_true", help="validate metadata, push nothing")
    parser.add_argument("--interval", type=int, default=20, help="seconds between status polls (default 20)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds before giving up polling (default 3600)")
    parser.add_argument("--log-file", type=Path, default=None, help="also write logs here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.log_file, args.verbose)

    kernel_dir = args.kernel_dir.resolve()
    try:
        meta = load_metadata(kernel_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        LOG.error("%s", exc)
        return 1

    LOG.info(
        "kernel '%s' (%s, gpu=%s, internet=%s)",
        meta["title"],
        meta["kernel_type"],
        meta.get("enable_gpu", False),
        meta.get("enable_internet", False),
    )
    if meta.get("dataset_sources"):
        LOG.info("attached datasets: %s", ", ".join(meta["dataset_sources"]))

    if args.dry_run:
        LOG.info("--dry-run: metadata is valid, nothing pushed")
        return 0

    try:
        creds = export_to_env()
    except CredentialsError as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info(creds.describe())

    kernel_id = resolve_id(meta, creds.username, kernel_dir)

    try:
        proc = run_kaggle(["kernels", "push", "-p", str(kernel_dir)])
    except KaggleCliError as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("pushed: %s", (proc.stdout or "").strip())
    published = published_ref(proc.stdout or "", kernel_id)
    if published != kernel_id:
        LOG.warning("Kaggle published to '%s', not the declared id '%s' -- it slugifies "
                    "the title. Polling the published ref.", published, kernel_id)
    kernel_id = published
    LOG.info("watch it at https://www.kaggle.com/code/%s", kernel_id)

    if args.no_wait:
        LOG.info("--no-wait: not polling. Check status with: kaggle kernels status %s", kernel_id)
        return 0

    status = poll_status(kernel_id, args.interval, args.timeout)
    if "error" in status.lower():
        LOG.error("kernel finished with an error; pull the log with pull_output.py")
        return 1

    LOG.info("done. Fetch output with: python kaggle/pull_output.py --kernel %s", kernel_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
