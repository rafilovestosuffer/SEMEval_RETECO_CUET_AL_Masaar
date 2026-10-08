"""The v2 rerank kernel resumes across Kaggle sessions without changing its output.

A full 1b split does not fit one 9 h GPU session. Session 2 gets session 1's runs as ``done/``
in its input; its runs must then equal what one uninterrupted session would have written.
Runs the kernel end to end with ``RERANK_FAKE=1`` (no model, no GPU, no network).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from reteco.paths import repo_root

KERNEL = repo_root() / "kaggle" / "kernels" / "v2_rerank" / "v2_rerank.py"
SOURCE = KERNEL.read_text(encoding="utf-8")
SPLIT = re.search(r'^SPLIT = "(\w+)"', SOURCE, re.M).group(1)
SUBTRACK = re.search(r'^SUBTRACK = "(\w+)"', SOURCE, re.M).group(1)
RUN_NAME = f"run_{SUBTRACK}_{SPLIT}.trec"


def _fixture(root: Path) -> tuple[Path, Path]:
    """Input dir (package + candidate runs) and data dir for one tiny domain."""
    inp, data = root / "in", root / "data" / "track1_tempo" / "dom"
    shutil.copytree(repo_root() / "reteco", inp / "reteco",
                    ignore=shutil.ignore_patterns("__pycache__"))
    data.mkdir(parents=True)
    docs = [f"d{i}" for i in range(40)]
    with (data / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"id": d, "content": f"text of {d}"}) + "\n")
    topics: list[str] = []
    with (data / f"examples_{SPLIT}.jsonl").open("w", encoding="utf-8") as ex, \
            (data / f"steps_{SPLIT}.jsonl").open("w", encoding="utf-8") as st:
        for q in range(3):
            qid = f"q{q}"
            steps = [{"step_id": f"{qid}_step{s}", "step": f"s{s}",
                      "step_instruction": f"instruction {s}"} for s in range(1, 3)]
            ex.write(json.dumps({"id": qid, "query": f"query {q}"}) + "\n")
            st.write(json.dumps({"id": qid, "query": f"query {q}", "steps": steps}) + "\n")
            topics += [qid] if SUBTRACK == "1a" else [s["step_id"] for s in steps]
    run = inp / "runs" / "dom" / RUN_NAME
    run.parent.mkdir(parents=True)
    with run.open("w", encoding="utf-8") as fh:
        for t_i, t in enumerate(topics):
            ranked = docs[t_i:] + docs[:t_i]
            for r, d in enumerate(ranked, 1):
                fh.write(f"{t} Q0 {d} {r} {100 - r} dense\n")
    return inp, root / "data"


def _run(work: Path, inp: Path, data: Path, limit: int = 0) -> dict:
    work.mkdir(parents=True)
    env = dict(os.environ, RERANK_FAKE="1", RERANK_INPUT=str(inp), RERANK_DATA=str(data),
               RERANK_LIMIT=str(limit), PYTHONUTF8="1")
    proc = subprocess.run([sys.executable, str(KERNEL)], cwd=work, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads((work / "p7_out" / "v2_rerank_report.json").read_text(encoding="utf-8"))


def test_two_sessions_equal_one(tmp_path: Path) -> None:
    inp, data = _fixture(tmp_path)
    full = _run(tmp_path / "full", inp, data)
    first = _run(tmp_path / "s1", inp, data, limit=2)
    assert first["queries_done"] == 2

    shutil.copytree(tmp_path / "s1" / "p7_out" / "runs", inp / "done")
    second = _run(tmp_path / "s2", inp, data)

    assert second["queries_resumed"] == 2
    assert second["queries_done"] == full["queries_done"] == full["queries_planned"]
    assert second["stats"]["windows"] < full["stats"]["windows"]   # resumed topics not redone
    out_full = (tmp_path / "full" / "p7_out" / "runs" / "dom" / RUN_NAME).read_text("utf-8")
    out_two = (tmp_path / "s2" / "p7_out" / "runs" / "dom" / RUN_NAME).read_text("utf-8")
    assert out_two == out_full
