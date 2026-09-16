"""Make both the `reteco` package and the `kaggle/` scripts importable from tests.

`kaggle/` is not a package (adding __init__.py there would shadow the installed kaggle
library from the repo root), so its directory goes on sys.path directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for entry in (REPO_ROOT, REPO_ROOT / "kaggle"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
