#!/usr/bin/env python3
"""Wrapper around the organizers' `format_checker.py`.

CLAUDE.md §4: "Always run `format_checker.py` with `--qrels` and `--corpus` before scoring,"
and its own docstring says a submission that fails validation should never reach the
competition platform.

The checker lives in the RETECO starter kit, which we do **not** vendor — that repo has no
LICENSE file (`notes/starter_kit_findings.md` §5). It is cloned at the pinned commit when
needed. Set `RETECO_STARTER_KIT` to an existing clone to skip the network entirely, which is
how this runs on a machine with no internet.

A fallback validator implements the same documented rules for the case where the starter kit
is unreachable. It is a safety net, not a substitute: the organizers' checker is the authority,
and `validate` says which one ran.

Usage::

    python submit/check_format.py run.txt --qrels qrels.txt --corpus documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reteco.paths import cache_root  # noqa: E402

__all__ = ["KIT_COMMIT", "CheckResult", "find_starter_kit", "validate", "fallback_validate"]

KIT_URL = "https://github.com/DataScienceUIBK/RETECO.git"
KIT_COMMIT = "23093c30e568a337c6293de4c58ffcccddbded5b"
EXPECTED_COLUMNS = 6


@dataclass
class CheckResult:
    """Outcome of a format check."""

    valid: bool
    checker: str  # "official" or "fallback"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    num_lines: int = 0
    num_topics: int = 0
    output: str = ""

    def summary(self) -> str:
        head = (f"[{self.checker}] lines: {self.num_lines}  topics: {self.num_topics}  "
                f"errors: {len(self.errors)}  warnings: {len(self.warnings)}")
        body = "\n".join(f"  ERROR   {e}" for e in self.errors[:50])
        warn = "\n".join(f"  warn    {w}" for w in self.warnings[:20])
        tail = "RESULT: VALID" if self.valid else "RESULT: INVALID"
        return "\n".join(p for p in (head, body, warn, tail) if p)


def find_starter_kit(auto_clone: bool = True) -> Path | None:
    """Locate `starter_kit/`, from the env var or a cached clone at the pinned commit."""
    override = os.environ.get("RETECO_STARTER_KIT")
    if override:
        candidate = Path(override).expanduser()
        for path in (candidate, candidate / "starter_kit"):
            if (path / "format_checker.py").is_file():
                return path
        return None

    cached = cache_root() / "reteco_starter_kit"
    if (cached / "starter_kit" / "format_checker.py").is_file():
        return cached / "starter_kit"
    if not auto_clone:
        return None

    cached.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(["git", "clone", "--quiet", KIT_URL, str(cached)],
                           capture_output=True, text=True)
    if clone.returncode != 0:
        return None
    subprocess.run(["git", "-C", str(cached), "checkout", "--quiet", KIT_COMMIT],
                   capture_output=True, text=True)
    kit = cached / "starter_kit"
    return kit if (kit / "format_checker.py").is_file() else None


def fallback_validate(run: Path, qrels: Path | None = None, corpus: Path | None = None,
                      doc_key: str = "id") -> CheckResult:
    """Re-implementation of the documented rules, for when the starter kit is unreachable.

    Enforces exactly what `format_checker.py` documents: six whitespace-separated columns;
    rank a positive integer; score a float; ranks unique within a topic; warn when score
    increases with rank; warn on topic ids absent from qrels or doc ids absent from the corpus.
    """
    valid_topics = None
    if qrels is not None:
        valid_topics = {ln.split()[0] for ln in Path(qrels).read_text(encoding="utf-8").splitlines()
                        if ln.strip()}
    valid_docs = None
    if corpus is not None:
        valid_docs = {json.loads(ln)[doc_key]
                      for ln in Path(corpus).read_text(encoding="utf-8").splitlines() if ln.strip()}

    errors, warnings = [], []
    ranks_seen: dict[str, set[int]] = {}
    last_score: dict[str, float] = {}
    num_lines = 0

    for lineno, raw in enumerate(Path(run).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        num_lines += 1
        cols = line.split()
        if len(cols) != EXPECTED_COLUMNS:
            errors.append(f"L{lineno}: expected {EXPECTED_COLUMNS} columns, got {len(cols)}")
            continue
        topic_id, _q0, doc_id, rank_s, score_s, _tag = cols

        try:
            rank = int(rank_s)
            if rank <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"L{lineno}: rank must be a positive integer, got '{rank_s}'")
            rank = None
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"L{lineno}: score must be a float, got '{score_s}'")
            continue

        if valid_topics is not None and topic_id not in valid_topics:
            warnings.append(f"L{lineno}: topic '{topic_id}' not in qrels")
        if valid_docs is not None and doc_id not in valid_docs:
            warnings.append(f"L{lineno}: docid '{doc_id}' not in corpus")

        if rank is not None:
            seen = ranks_seen.setdefault(topic_id, set())
            if rank in seen:
                errors.append(f"L{lineno}: duplicate rank {rank} in topic '{topic_id}'")
            seen.add(rank)

        previous = last_score.get(topic_id)
        if previous is not None and score > previous + 1e-9:
            warnings.append(f"L{lineno}: score increases with rank in topic '{topic_id}'")
        last_score[topic_id] = score

    return CheckResult(valid=not errors, checker="fallback", errors=errors, warnings=warnings,
                       num_lines=num_lines, num_topics=len(ranks_seen))


def validate(run: Path, qrels: Path | None = None, corpus: Path | None = None,
             doc_key: str = "id", allow_fallback: bool = True) -> CheckResult:
    """Validate a run with the organizers' checker, falling back only if it is unavailable."""
    kit = find_starter_kit()
    if kit is None:
        if not allow_fallback:
            raise RuntimeError(
                "starter kit not found and fallback disabled. Set RETECO_STARTER_KIT to a "
                f"clone of {KIT_URL} at {KIT_COMMIT[:8]}.")
        return fallback_validate(run, qrels, corpus, doc_key)

    cmd = [sys.executable, str(kit / "format_checker.py"), str(run)]
    if qrels is not None:
        cmd += ["--qrels", str(qrels)]
    if corpus is not None:
        cmd += ["--corpus", str(corpus), "--doc-key", doc_key]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ""
    errors = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("ERROR")]
    warnings = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("warn")]

    num_lines = num_topics = 0
    for line in stdout.splitlines():
        if line.startswith("lines:"):
            parts = line.replace(":", " ").split()
            try:
                num_lines = int(parts[parts.index("lines") + 1])
                num_topics = int(parts[parts.index("topics") + 1])
            except (ValueError, IndexError):
                pass

    return CheckResult(valid=proc.returncode == 0, checker="official", errors=errors,
                       warnings=warnings, num_lines=num_lines, num_topics=num_topics,
                       output=stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--doc-key", default="id", choices=("id", "doc_id"))
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail rather than use the local re-implementation")
    args = parser.parse_args()

    result = validate(args.run, args.qrels, args.corpus, args.doc_key,
                      allow_fallback=not args.no_fallback)
    print(result.output.rstrip() if result.output else result.summary())
    if result.checker == "fallback":
        print("\nNOTE: the organizers' checker was unavailable; this used the local\n"
              "re-implementation. Re-check with the real one before submitting.")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
