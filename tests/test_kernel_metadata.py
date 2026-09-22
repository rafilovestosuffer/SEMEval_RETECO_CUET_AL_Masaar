"""Kernel metadata must be valid before a push burns quota or fails on the server."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import push_kernel
from reteco.paths import repo_root

KERNEL_DIRS = sorted(p for p in (repo_root() / "kaggle" / "kernels").iterdir() if p.is_dir())


@pytest.mark.parametrize("kernel_dir", KERNEL_DIRS, ids=lambda p: p.name)
def test_metadata_loads_and_names_an_existing_script(kernel_dir: Path) -> None:
    meta = push_kernel.load_metadata(kernel_dir)
    assert meta["language"] == "python"
    assert meta["kernel_type"] in {"script", "notebook"}
    assert (kernel_dir / meta["code_file"]).is_file()


@pytest.mark.parametrize("kernel_dir", KERNEL_DIRS, ids=lambda p: p.name)
def test_kernels_are_private_and_carry_no_credentials(kernel_dir: Path) -> None:
    raw = (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    meta = json.loads(raw)
    assert meta["is_private"] is True
    for forbidden in ("key", "token", "secret", "password"):
        assert forbidden not in {k.lower() for k in meta}, f"{forbidden} must never appear in kernel metadata"


def test_smoke_kernel_asks_for_a_gpu_and_no_internet() -> None:
    """The smoke test exists to prove GPU access; internet is off so it cannot silently download."""
    meta = json.loads((repo_root() / "kaggle" / "kernels" / "smoke_gpu" / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == []


def test_placeholder_is_substituted_and_written_back(tmp_path: Path) -> None:
    meta = {
        "id": "USERNAME/reteco-smoke-gpu",
        "title": "t",
        "code_file": "x.py",
        "language": "python",
        "kernel_type": "script",
    }
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    resolved = push_kernel.resolve_id(dict(meta), "rafiurrahman01", tmp_path)
    assert resolved == "rafiurrahman01/reteco-smoke-gpu"

    written = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert written["id"] == resolved


def test_a_real_id_is_left_alone(tmp_path: Path) -> None:
    meta = {"id": "someone/thing", "title": "t", "code_file": "x.py", "language": "python", "kernel_type": "script"}
    assert push_kernel.resolve_id(dict(meta), "rafiurrahman01", tmp_path) == "someone/thing"
    assert not (tmp_path / "kernel-metadata.json").exists(), "an unchanged id must not rewrite the file"


def test_missing_metadata_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kernel-metadata.json"):
        push_kernel.load_metadata(tmp_path)


def test_metadata_naming_a_missing_script_is_caught(tmp_path: Path) -> None:
    meta = {"id": "a/b", "title": "t", "code_file": "gone.py", "language": "python", "kernel_type": "script"}
    (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="gone.py"):
        push_kernel.load_metadata(tmp_path)


def slugify(title: str) -> str:
    """Kaggle's slug rule: lowercase, non-alphanumerics collapse to single hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@pytest.mark.parametrize("kernel_dir", KERNEL_DIRS, ids=lambda p: p.name)
def test_title_slugifies_to_the_id(kernel_dir: Path) -> None:
    """Kaggle derives the real URL slug from the TITLE, not from the id we declare.

    Phase 2's first push proved it: title "RETECO phase2 BM25 full track1" published to
    .../reteco-phase2-bm25-full-track1 while the metadata said reteco-phase2-bm25-full.
    The kernel ran fine and `kernels status` on the declared id returned "Permission
    'kernels.get' was denied", which reads like an auth problem and is not one. Keeping
    the two in sync is what stops a push from polling a kernel that does not exist.
    """
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    slug = meta["id"].split("/", 1)[1]
    assert slugify(meta["title"]) == slug, (
        f"title {meta['title']!r} publishes to slug {slugify(meta['title'])!r}, "
        f"but the metadata id says {slug!r}"
    )


def test_published_ref_prefers_the_url_kaggle_printed() -> None:
    out = ("Kernel version 1 successfully pushed.  Please check progress at "
           "https://www.kaggle.com/code/rafiurrahman01/reteco-phase2-bm25-full-track1")
    assert push_kernel.published_ref(out, "rafiurrahman01/reteco-phase2-bm25-full") == \
        "rafiurrahman01/reteco-phase2-bm25-full-track1"


def test_published_ref_falls_back_when_no_url_is_printed() -> None:
    assert push_kernel.published_ref("no url here", "user/slug") == "user/slug"
