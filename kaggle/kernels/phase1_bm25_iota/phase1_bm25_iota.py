"""Phase 1 — reproduce the organizers' official BM25 baseline on the IOTA domain.

CPU kernel, internet ON, no GPU (costs zero GPU quota). CLAUDE.md §9, Phase 1 gate:
our numbers must match starter_kit/BASELINE_RESULTS.md for IOTA to 4 decimals.

Runs in two stages so a single push is always informative:

  STAGE A  probe the ground we cannot see from a Claude Code session:
             - is there a JDK new enough for pyserini?
             - what is the actual file layout of the HF dataset repo?
           If either is wrong, stop with a one-line diagnosis, not a stack trace.

  STAGE B  reproduce:
             - download ONLY track1_tempo/iota (~10k docs), not the 1.65M-doc corpus
             - clone the starter kit at the pinned commit and run it verbatim
             - compare against the published figures and print PASS/FAIL

Why invoke their code instead of reimplementing BM25: a reimplementation that matches
proves nothing their own code doesn't already prove, and silently diverges later. The
RETECO repo has no LICENSE file, so it is cloned at runtime, never vendored.

Everything is written to /kaggle/working as it is produced, so a killed session still
returns the probe output.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
DATA_DIR = WORKING / "reteco_data"
KIT_DIR = WORKING / "reteco_starter_kit"
OUT_DIR = WORKING / "baseline_out"
REPORT = WORKING / "phase1_report.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
KIT_URL = "https://github.com/DataScienceUIBK/RETECO.git"
KIT_COMMIT = "23093c30e568a337c6293de4c58ffcccddbded5b"  # 2026-08-30
DOMAIN = "iota"
# pyserini's bundled Lucene jars are compiled for Java 21 (class file version 65.0).
# Measured, not assumed: run 1 of this kernel found Kaggle's default JDK 17 (class file
# version 61.0) and died with UnsupportedClassVersionError on
# org/apache/lucene/analysis/Analyzer. 11 was too lenient -- it let 17 pass Stage A and
# pushed the failure into Stage B where the diagnosis was a Java stack trace.
MIN_JDK = 21

# starter_kit/BASELINE_RESULTS.md @ 23093c3 -- the ORGANIZERS' published figures.
PUBLISHED = {"1a_train": 0.0199, "1a_dev": 0.2083, "1b_train": 0.0000, "1b_dev": 0.3289}
# A published 0.0000 is not evidence: a broken pipeline reproduces it by accident.
LOAD_BEARING = [k for k, v in PUBLISHED.items() if v != 0.0]
TOLERANCE = 5e-5  # observed must ROUND to the published 4-dp value

report: dict[str, object] = {"stage": "A", "kit_commit": KIT_COMMIT, "hf_repo": HF_REPO}


def save_report() -> None:
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, **kw)


def die(reason: str, code: int = 1) -> None:
    """Stop with a diagnosis rather than a traceback."""
    report["blocked_by"] = reason
    save_report()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    print(f"\nreport written to {REPORT}", flush=True)
    raise SystemExit(code)


# ============================================================== STAGE A: probe ==
def probe_java() -> int | None:
    """Return the major JDK version visible to pyserini, or None."""
    exe = shutil.which("java")
    if exe is None:
        print("java: not on PATH")
        return None
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True)
    banner = (proc.stderr or proc.stdout or "").strip()
    print(banner)
    # 'openjdk version "21.0.10"' / '"1.8.0_402"' -> 21 / 8
    match = re.search(r'version "(\d+)(?:\.(\d+))?', banner)
    if not match:
        return None
    major = int(match.group(1))
    if major == 1 and match.group(2):  # legacy 1.8 style
        major = int(match.group(2))
    return major


def install_jdk() -> int | None:
    """Best-effort JDK install on the Kaggle image."""
    print("attempting to install a JDK ...", flush=True)
    for cmd in (["apt-get", "-qq", "update"],
                ["apt-get", "-qq", "-y", "install", "openjdk-21-jdk-headless"]):
        proc = run(cmd, capture_output=True)
        if proc.returncode != 0:
            print(f"  failed: {(proc.stderr or '')[-400:]}")
            return None
    return probe_java()


def find_jdk_home(minimum: int = MIN_JDK) -> tuple[int, Path] | None:
    """Highest installed JDK >= ``minimum`` under /usr/lib/jvm, as ``(major, home)``.

    Deliberately does NOT use ``shutil.which("java")``: apt can install openjdk-21
    while update-alternatives leaves /usr/bin/java pointing at 17, so the launcher on
    PATH is not evidence about which JDKs exist. Each candidate is interrogated by
    running its own ``bin/java -version``.
    """
    jvm_root = Path("/usr/lib/jvm")
    if not jvm_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for home in sorted(jvm_root.iterdir()):
        launcher = home / "bin" / "java"
        if not launcher.is_file():
            continue
        proc = subprocess.run([str(launcher), "-version"], capture_output=True, text=True)
        match = re.search(r'version "(\d+)(?:\.(\d+))?', proc.stderr or proc.stdout or "")
        if not match:
            continue
        major = int(match.group(1))
        if major == 1 and match.group(2):  # legacy 1.8 style
            major = int(match.group(2))
        if major >= minimum and (best is None or major > best[0]):
            best = (major, home)
    return best


def activate_jdk(home: Path) -> None:
    """Point pyserini at ``home``: JAVA_HOME, JVM_PATH, and PATH all agree.

    PATH is prepended too, so any subprocess that resolves ``java`` itself gets this
    JDK rather than whatever update-alternatives left in /usr/bin.
    """
    os.environ["JAVA_HOME"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    print(f"JAVA_HOME={home}")
    for candidate in home.rglob("libjvm.so"):
        os.environ["JVM_PATH"] = str(candidate)
        print(f"JVM_PATH={candidate}")
        break


def probe_hf_layout() -> list[str]:
    """List the dataset repo's files and print the tree.

    This single output settles the layout question for every later phase -- it cannot
    be checked from a Claude Code session, where huggingface.co is blocked.
    """
    from huggingface_hub import list_repo_files

    files = list_repo_files(HF_REPO, repo_type="dataset")
    print(f"{len(files)} files in {HF_REPO}\n")

    tops: dict[str, int] = {}
    for f in files:
        tops[f.split("/")[0]] = tops.get(f.split("/")[0], 0) + 1
    print("top level:")
    for name, count in sorted(tops.items()):
        print(f"  {name:<28} {count:>6} file(s)")

    domain_files = [f for f in files if f"{DOMAIN}/" in f]
    print(f"\nfiles matching '{DOMAIN}/' ({len(domain_files)}):")
    for f in sorted(domain_files)[:40]:
        print(f"  {f}")

    report["hf_file_count"] = len(files)
    report["hf_top_level"] = tops
    report["iota_files"] = sorted(domain_files)[:40]
    return files


def resolve_iota_prefix(files: list[str]) -> str:
    """Find the directory prefix holding the IOTA domain, whatever it is called."""
    expected = f"track1_tempo/{DOMAIN}"
    for f in files:
        if f.startswith(expected + "/"):
            return expected
    # Fall back to any directory containing a documents.jsonl next to the domain name.
    for f in files:
        if f.endswith(f"{DOMAIN}/documents.jsonl"):
            return f.rsplit("/", 1)[0]
    die(f"no '{expected}/' in {HF_REPO}; the printed tree above shows the real layout. "
        f"Update DOMAIN/prefix in this kernel and re-push.")
    raise AssertionError("unreachable")


# ========================================================= STAGE B: reproduce ==
def download(prefix: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"downloading {prefix}/* from {HF_REPO} ...", flush=True)
    started = time.monotonic()
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=[f"{prefix}/*"], max_workers=8)
    elapsed = time.monotonic() - started
    got = DATA_DIR / prefix
    files = sorted(p.name for p in got.iterdir()) if got.is_dir() else []
    print(f"  {len(files)} file(s) in {got} ({elapsed:.1f}s): {', '.join(files)}")
    report["downloaded_files"] = files
    report["download_seconds"] = round(elapsed, 1)
    if not files:
        die(f"download produced nothing under {got}")
    return got


def clone_kit() -> Path:
    if KIT_DIR.exists():
        shutil.rmtree(KIT_DIR)
    if run(["git", "clone", "--quiet", KIT_URL, str(KIT_DIR)]).returncode != 0:
        die(f"could not clone {KIT_URL}")
    if run(["git", "-C", str(KIT_DIR), "checkout", "--quiet", KIT_COMMIT]).returncode != 0:
        die(f"could not check out pinned commit {KIT_COMMIT[:8]}")
    head = subprocess.run(["git", "-C", str(KIT_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"starter kit at {head}")
    if head != KIT_COMMIT:
        die(f"pin mismatch: expected {KIT_COMMIT}, got {head}")
    return KIT_DIR / "starter_kit"


def install_deps() -> None:
    # pytrec-eval-terrier, NOT pytrec_eval: the latter compiles trec_eval from source
    # and fetches it over the network during the build, which fails behind any egress
    # restriction. terrier ships wheels and imports as `pytrec_eval` with the same API
    # -- verified locally on 2026-09-16 (pytrec_eval 0.5.10 via terrier).
    # pyserini is large (it pulls torch/faiss), hence the generous timeout and retries.
    proc = run([sys.executable, "-m", "pip", "install", "-q",
                "--timeout", "120", "--retries", "5",
                "pyserini", "gensim", "pytrec-eval-terrier", "tqdm"], capture_output=True)
    if proc.returncode != 0:
        die(f"pip install failed:\n{(proc.stderr or '')[-1500:]}")
    for module in ("pyserini", "gensim", "pytrec_eval"):
        got = subprocess.run(
            [sys.executable, "-c",
             f"import {module}; print(getattr({module}, '__version__', 'n/a'))"],
            capture_output=True, text=True)
        version = got.stdout.strip() or got.stderr.strip()[-200:]
        print(f"  {module:<12} {version}")
        report.setdefault("versions", {})[module] = version  # type: ignore[union-attr]
        if got.returncode != 0:
            die(f"{module} installed but will not import: {got.stderr.strip()[-400:]}")


def run_baseline(kit: Path) -> dict:
    """Invoke official_baseline.py verbatim. '--track2' with no values skips Track 2."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = run([sys.executable, str(kit / "official_baseline.py"),
                "--data", str(DATA_DIR), "--out", str(OUT_DIR),
                "--track1", DOMAIN, "--track2", "--splits", "train", "dev"],
               cwd=str(kit))
    elapsed = time.monotonic() - started
    report["baseline_seconds"] = round(elapsed, 1)
    print(f"\nofficial_baseline.py exited {proc.returncode} after {elapsed:.1f}s")

    results = OUT_DIR / "track1_tempo" / DOMAIN / "results.json"
    if not results.is_file():
        die(f"official_baseline.py produced no {results} (exit {proc.returncode}); "
            f"see the traceback above")
    return json.loads(results.read_text(encoding="utf-8"))


def gate(results: dict) -> bool:
    """Compare against the published IOTA figures. Zero-valued rows abstain."""
    print(f"\n  {'key':<10} {'published':>10} {'observed':>10} {'delta':>10} "
          f"{'topics':>7}  status")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 7}  ------")

    passed = True
    rows = {}
    for key, expected in sorted(PUBLISHED.items()):
        entry = results.get(key) or {}
        observed = entry.get("NDCG@10")
        topics = entry.get("num_topics", "?")
        counts = key in LOAD_BEARING
        if observed is None:
            status, shown, delta = "MISSING", "     -    ", "     -    "
            ok = False
        else:
            ok = abs(observed - expected) < TOLERANCE
            status = ("match" if counts else "match (not counted)") if ok else "MISMATCH"
            shown, delta = f"{observed:10.4f}", f"{observed - expected:+10.4f}"
        if counts and not ok:
            passed = False
        rows[key] = {"published": expected, "observed": observed,
                     "num_topics": topics, "counts": counts, "matched": ok}
        print(f"  {key:<10} {expected:10.4f} {shown} {delta} {str(topics):>7}  {status}")

    report["gate_rows"] = rows
    report["gate_passed"] = passed
    print(f"\n  Not counted: 1b_train — published as 0.0000, so a match proves nothing")
    print( "  (a broken pipeline returns 0.0000 too). Load-bearing: "
          f"{', '.join(sorted(LOAD_BEARING))}.")
    print( "  IOTA is ~10 queries split 70/30, so dev is ~3 topics. This is wiring")
    print( "  evidence, not performance evidence.")
    return passed


# ============================================================================ --
def main() -> int:
    section(f"STAGE A.1 — JDK (pyserini's Lucene jars need {MIN_JDK}+)")
    report["default_jdk"] = probe_java()  # what's on PATH, for the record
    found = find_jdk_home()
    if found is None:
        print(f"no installed JDK >= {MIN_JDK}; installing one")
        install_jdk()
        found = find_jdk_home()
        if found is None:
            die(f"no JDK >= {MIN_JDK} available after install. pyserini's Lucene analyzer "
                f"is compiled for Java {MIN_JDK} (class file version 65.0) and cannot run "
                f"on an older runtime, so the official numbers are unreachable.")
    major, home = found
    activate_jdk(home)
    report["jdk_major"] = major
    report["java_home"] = str(home)
    print(f"JDK {major} OK")
    save_report()

    section(f"STAGE A.2 — layout of {HF_REPO}")
    try:
        files = probe_hf_layout()
    except Exception as exc:  # noqa: BLE001 - any failure here is a clean stop
        die(f"could not list {HF_REPO}: {type(exc).__name__}: {exc}. "
            f"Is internet enabled on this kernel?")
    prefix = resolve_iota_prefix(files)
    report["iota_prefix"] = prefix
    print(f"\nresolved IOTA prefix: {prefix}")
    save_report()

    report["stage"] = "B"
    section("STAGE B.1 — download IOTA only")
    download(prefix)
    save_report()

    section(f"STAGE B.2 — starter kit @ {KIT_COMMIT[:8]}")
    kit = clone_kit()
    save_report()

    section("STAGE B.3 — dependencies")
    install_deps()
    save_report()

    section("STAGE B.4 — official_baseline.py")
    results = run_baseline(kit)
    report["results"] = results
    save_report()

    section("STAGE B.5 — Phase 1 gate vs BASELINE_RESULTS.md @ 23093c3")
    passed = gate(results)
    save_report()

    section(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"report:  {REPORT}")
    print(f"runs:    {OUT_DIR / 'track1_tempo' / DOMAIN}")
    if passed:
        print("\nPhase 1 gate closed. Log the ledger row, then propose Phase 2.")
    else:
        print("\nGate not closed. The per-key deltas above say which sub-track diverged.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
