"""Phase 2 — full Track 1 BM25 reproduction + data audit (CLAUDE.md §9).

CPU kernel, internet ON, no GPU. Two deliverables:

  1. **Reproduction.** Run the organizers' `official_baseline.py` over all 13 TEMPO
     domains and compare per-domain and macro nDCG@10 against BASELINE_RESULTS.md
     @ 23093c3. The macro is the equal-weight mean over domains, never over topics.

  2. **Audit.** Everything CLAUDE.md §9 Phase 2 asks for -- document length
     distribution, query/step length, gold per query, duplicates, doc id formats --
     plus the one question that could change the shape of the paper: **does
     `guidance_*.jsonl` carry TEMPO's reasoning-class labels (TCP, HAC, CAU...)?**
     If it does, a per-reasoning-class breakdown beats a per-domain one and H6 is
     answerable (`notes/lit/SUMMARY.md`).

Stage order is deliberate: **the audit runs before the baseline.** The audit is pure
Python over files already on disk and takes seconds; the baseline indexes 1.65M
documents and takes roughly an hour. Running the cheap, novel-information stage first
means a session that dies in the expensive stage still returns the audit.

Phase 1 (commit fa495d3) established two things this kernel depends on: the HF layout
is `track1_tempo/<domain>/*`, and Kaggle's default JDK 17 cannot run pyserini's Lucene
jars (class file version 65.0 = Java 21). The JDK selection below is carried over
verbatim -- it installs 21 and points JAVA_HOME/JVM_PATH/PATH at it explicitly, because
apt does not necessarily repoint /usr/bin/java.

Published figures are inlined rather than imported: a Kaggle kernel is pushed as a lone
script and cannot see `eval/gate.py`. The authoritative verdict is still the local one --
pull the results and run `python eval/gate.py --all`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
DATA_DIR = WORKING / "reteco_data"
KIT_DIR = WORKING / "reteco_starter_kit"
OUT_DIR = WORKING / "baseline_out"
REPORT = WORKING / "phase2_report.json"
AUDIT = WORKING / "data_audit.json"

HF_REPO = "DataScience-UIBK/RETECO-SemEval2027"
KIT_URL = "https://github.com/DataScienceUIBK/RETECO.git"
KIT_COMMIT = "23093c30e568a337c6293de4c58ffcccddbded5b"  # 2026-08-30
TRACK = "track1_tempo"
MIN_JDK = 21  # pyserini's Lucene jars are class file version 65.0; measured in Phase 1.

# starter_kit/BASELINE_RESULTS.md @ 23093c3 -- the ORGANIZERS' figures, not ours.
PUBLISHED: dict[str, dict[str, float]] = {
    "iota": {"1a_train": 0.0199, "1a_dev": 0.2083, "1b_train": 0.0000, "1b_dev": 0.3289},
    "bitcoin": {"1a_train": 0.0695, "1a_dev": 0.0263, "1b_train": 0.0774, "1b_dev": 0.0205},
    "cardano": {"1a_train": 0.1349, "1a_dev": 0.0851, "1b_train": 0.1174, "1b_dev": 0.0554},
    "economics": {"1a_train": 0.0382, "1a_dev": 0.0480, "1b_train": 0.0278, "1b_dev": 0.0517},
    "genealogy": {"1a_train": 0.1003, "1a_dev": 0.1677, "1b_train": 0.0982, "1b_dev": 0.2060},
    "history": {"1a_train": 0.0691, "1a_dev": 0.0877, "1b_train": 0.0651, "1b_dev": 0.0989},
    "hsm": {"1a_train": 0.1627, "1a_dev": 0.2239, "1b_train": 0.1591, "1b_dev": 0.2084},
    "law": {"1a_train": 0.0943, "1a_dev": 0.0574, "1b_train": 0.0846, "1b_dev": 0.0549},
    "monero": {"1a_train": 0.0278, "1a_dev": 0.0252, "1b_train": 0.0517, "1b_dev": 0.0103},
    "politics": {"1a_train": 0.2792, "1a_dev": 0.2550, "1b_train": 0.2425, "1b_dev": 0.2306},
    "quant": {"1a_train": 0.0255, "1a_dev": 0.0218, "1b_train": 0.0085, "1b_dev": 0.0374},
    "travel": {"1a_train": 0.0429, "1a_dev": 0.0275, "1b_train": 0.0378, "1b_dev": 0.0492},
    "workplace": {"1a_train": 0.0777, "1a_dev": 0.0230, "1b_train": 0.1369, "1b_dev": 0.0302},
}
PUBLISHED_MACRO = {"1a_train": 0.0879, "1a_dev": 0.0967, "1b_train": 0.0852, "1b_dev": 0.1063}
KEYS = ("1a_train", "1a_dev", "1b_train", "1b_dev")
TOLERANCE = 5e-5  # observed must ROUND to the published 4-dp figure

# TEMPO's reasoning classes. Phase 0 could not confirm whether the release ships them;
# `audit_guidance` looks for these tokens in every guidance field (notes/lit/tempo.md).
REASONING_CLASSES = ["TCP", "HAC", "CAU", "CMP", "SEQ", "TRN", "AGG", "DUR"]

report: dict[str, object] = {"stage": "A", "kit_commit": KIT_COMMIT, "hf_repo": HF_REPO}


def save_report() -> None:
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, **kw)


def die(reason: str, code: int = 1) -> None:
    report["blocked_by"] = reason
    save_report()
    print(f"\n{'!' * 72}\nBLOCKED: {reason}\n{'!' * 72}", flush=True)
    print(f"\nreport written to {REPORT}", flush=True)
    raise SystemExit(code)


# ================================================================ JDK (Phase 1) ==
def probe_java() -> int | None:
    exe = shutil.which("java")
    if exe is None:
        print("java: not on PATH")
        return None
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True)
    banner = (proc.stderr or proc.stdout or "").strip()
    print(banner)
    match = re.search(r'version "(\d+)(?:\.(\d+))?', banner)
    if not match:
        return None
    major = int(match.group(1))
    if major == 1 and match.group(2):
        major = int(match.group(2))
    return major


def install_jdk() -> None:
    print("installing openjdk-21 ...", flush=True)
    for cmd in (["apt-get", "-qq", "update"],
                ["apt-get", "-qq", "-y", "install", "openjdk-21-jdk-headless"]):
        proc = run(cmd, capture_output=True)
        if proc.returncode != 0:
            print(f"  failed: {(proc.stderr or '')[-400:]}")
            return


def find_jdk_home(minimum: int = MIN_JDK) -> tuple[int, Path] | None:
    """Highest installed JDK >= ``minimum``, by running each candidate's own launcher.

    Not ``shutil.which("java")``: apt can install 21 while update-alternatives leaves
    /usr/bin/java on 17. Kaggle installs 21 as java-1.21.0-openjdk-amd64, so the
    directory name is not a reliable version either.
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
        if major == 1 and match.group(2):
            major = int(match.group(2))
        if major >= minimum and (best is None or major > best[0]):
            best = (major, home)
    return best


def activate_jdk(home: Path) -> None:
    os.environ["JAVA_HOME"] = str(home)
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    print(f"JAVA_HOME={home}")
    for candidate in home.rglob("libjvm.so"):
        os.environ["JVM_PATH"] = str(candidate)
        print(f"JVM_PATH={candidate}")
        break


# ==================================================================== download ==
def download() -> Path:
    from huggingface_hub import snapshot_download

    print(f"downloading {TRACK}/* from {HF_REPO} (all 13 domains) ...", flush=True)
    started = time.monotonic()
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATA_DIR),
                      allow_patterns=[f"{TRACK}/*"], max_workers=8)
    elapsed = time.monotonic() - started
    root = DATA_DIR / TRACK
    domains = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"  {len(domains)} domains, {total / 1e9:.2f} GB, {elapsed:.1f}s")
    print(f"  {', '.join(domains)}")
    report["domains"] = domains
    report["download_seconds"] = round(elapsed, 1)
    report["corpus_bytes"] = total
    if not domains:
        die(f"download produced no domain directories under {root}")
    missing = sorted(set(PUBLISHED) - set(domains))
    if missing:
        print(f"  NOTE: published table names domains not present: {missing}")
    return root


# ======================================================================= audit ==
def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarise(values: list[int]) -> dict[str, float]:
    """Distribution summary. Percentiles decide truncation and chunking in Phase 4."""
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "p25": pct(0.25), "median": pct(0.50),
            "p75": pct(0.75), "p90": pct(0.90), "p99": pct(0.99), "max": ordered[-1],
            "mean": round(statistics.fmean(ordered), 1)}


def audit_guidance(domain_dir: Path, split: str) -> dict:
    """What is actually inside guidance_{split}.jsonl, and does it name a reasoning class?

    Answers H6's precondition. Records every key seen, the shape of each value, and --
    for short string/list fields -- the value counts, since that is how a class label
    would present. Also scans all text for TEMPO's class tokens.
    """
    path = domain_dir / f"guidance_{split}.jsonl"
    if not path.is_file():
        return {"present": False}

    keys: Counter = Counter()
    value_kinds: dict[str, Counter] = {}
    small_values: dict[str, Counter] = {}
    class_hits: Counter = Counter()
    records = 0
    sample = None

    for record in read_jsonl(path):
        records += 1
        if sample is None:
            sample = {k: (v if not isinstance(v, (list, dict)) else type(v).__name__)
                      for k, v in record.items()}
        for key, value in record.items():
            keys[key] += 1
            value_kinds.setdefault(key, Counter())[type(value).__name__] += 1
            if isinstance(value, str) and len(value) <= 40:
                small_values.setdefault(key, Counter())[value] += 1
            elif isinstance(value, list) and all(
                    isinstance(x, str) and len(x) <= 40 for x in value):
                for item in value:
                    small_values.setdefault(key, Counter())[item] += 1
        blob = json.dumps(record)
        for token in REASONING_CLASSES:
            if re.search(rf"\b{token}\b", blob):
                class_hits[token] += 1

    return {
        "present": True,
        "records": records,
        "keys": dict(keys),
        "value_types": {k: dict(v) for k, v in value_kinds.items()},
        "small_value_counts": {k: dict(v.most_common(12))
                               for k, v in small_values.items() if len(v) <= 40},
        "reasoning_class_hits": dict(class_hits),
        "sample_record_shape": sample,
    }


def audit_domain(domain: str, domain_dir: Path) -> dict:
    """Everything CLAUDE.md §9 Phase 2 asks about one domain."""
    out: dict[str, object] = {"domain": domain}

    # --- corpus: lengths, duplicate ids, duplicate text, id format ---------------
    char_lens: list[int] = []
    word_lens: list[int] = []
    ids: Counter = Counter()
    content_hashes: Counter = Counter()
    id_samples: list[str] = []
    empty_docs = 0
    for record in read_jsonl(domain_dir / "documents.jsonl"):
        doc_id = str(record.get("id", record.get("doc_id", "")))
        ids[doc_id] += 1
        if len(id_samples) < 5:
            id_samples.append(doc_id)
        text = record.get("content") or ""
        if not text.strip():
            empty_docs += 1
        char_lens.append(len(text))
        word_lens.append(len(text.split()))
        content_hashes[hashlib.md5(text.encode("utf-8")).hexdigest()] += 1

    duplicate_ids = {k: v for k, v in ids.items() if v > 1}
    duplicate_text = sum(v - 1 for v in content_hashes.values() if v > 1)
    corpus_ids = set(ids)
    out["corpus"] = {
        "n_docs": sum(ids.values()),
        "n_unique_ids": len(ids),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_text_docs": duplicate_text,
        "empty_docs": empty_docs,
        "doc_chars": summarise(char_lens),
        "doc_words": summarise(word_lens),
        "id_samples": id_samples,
        "id_all_numeric": all(s.isdigit() for s in id_samples),
    }

    # --- topics: query/step lengths, gold counts, unreachable gold ---------------
    for split in ("train", "dev"):
        examples = domain_dir / f"examples_{split}.jsonl"
        if examples.is_file():
            q_words, gold_counts, unreachable, no_gold = [], [], 0, 0
            for record in read_jsonl(examples):
                q_words.append(len((record.get("query") or "").split()))
                gold = list(record.get("gold_ids") or [])
                gold_counts.append(len(gold))
                missing = [g for g in gold if g not in corpus_ids]
                unreachable += len(missing)
                if gold and len(missing) == len(gold):
                    no_gold += 1
            out[f"1a_{split}"] = {
                "n_queries": len(q_words),
                "query_words": summarise(q_words),
                "gold_per_query": summarise(gold_counts),
                "unreachable_gold_ids": unreachable,
                "queries_with_all_gold_unreachable": no_gold,
                "scoreable_queries": len(q_words) - no_gold,
            }

        steps_file = domain_dir / f"steps_{split}.jsonl"
        if steps_file.is_file():
            per_query, instr_words, step_gold, s_unreachable, s_no_gold = [], [], [], 0, 0
            for record in read_jsonl(steps_file):
                steps = record.get("steps") or []
                per_query.append(len(steps))
                for step in steps:
                    instr_words.append(len((step.get("step_instruction") or "").split()))
                    gold = list(step.get("gold_ids") or [])
                    step_gold.append(len(gold))
                    missing = [g for g in gold if g not in corpus_ids]
                    s_unreachable += len(missing)
                    if gold and len(missing) == len(gold):
                        s_no_gold += 1
            out[f"1b_{split}"] = {
                "n_queries": len(per_query),
                "n_steps": len(instr_words),
                "steps_per_query": summarise(per_query),
                "step_instruction_words": summarise(instr_words),
                "gold_per_step": summarise(step_gold),
                "unreachable_gold_ids": s_unreachable,
                "steps_with_all_gold_unreachable": s_no_gold,
                "scoreable_steps": len(instr_words) - s_no_gold,
            }

        out[f"guidance_{split}"] = audit_guidance(domain_dir, split)

    return out


def run_audit(root: Path, domains: list[str]) -> dict:
    audit: dict[str, object] = {"track": TRACK, "domains": {}}
    for domain in domains:
        started = time.monotonic()
        entry = audit_domain(domain, root / domain)
        audit["domains"][domain] = entry  # type: ignore[index]
        corpus = entry["corpus"]  # type: ignore[index]
        print(f"  {domain:<12} {corpus['n_docs']:>7} docs  "
              f"median {corpus['doc_words']['median']:>5} words  "
              f"p99 {corpus['doc_words']['p99']:>6}  "
              f"dupIDs {corpus['duplicate_ids']:>3}  "
              f"dupText {corpus['duplicate_text_docs']:>5}  "
              f"({time.monotonic() - started:.1f}s)", flush=True)
        AUDIT.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    return audit


def report_guidance(audit: dict) -> None:
    """The H6 precondition, stated plainly -- it decides a paper axis."""
    keys: Counter = Counter()
    classes: Counter = Counter()
    present = 0
    for entry in audit["domains"].values():
        for split in ("train", "dev"):
            guide = entry.get(f"guidance_{split}", {})
            if not guide.get("present"):
                continue
            present += 1
            keys.update(guide.get("keys", {}))
            classes.update(guide.get("reasoning_class_hits", {}))
    print(f"\n  guidance files present: {present}")
    print(f"  fields seen: {', '.join(sorted(keys)) or '(none)'}")
    if classes:
        print(f"  TEMPO reasoning-class tokens found: {dict(classes)}")
        print("  -> H6 is answerable from the release; consider a per-class breakdown.")
    else:
        print("  TEMPO reasoning-class tokens: NONE found")
        print("  -> H6 cannot be answered from guidance alone; per-domain stays the axis.")


# ==================================================================== baseline ==
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


def run_baseline(kit: Path, domains: list[str]) -> dict[str, dict]:
    """Invoke official_baseline.py over every domain. It caches per-domain results.json.

    Output is streamed (not captured) so a long run is visible in the Kaggle log, and so
    a session killed at the 12-hour limit still leaves the finished domains on disk.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = run([sys.executable, str(kit / "official_baseline.py"),
                "--data", str(DATA_DIR), "--out", str(OUT_DIR),
                "--track1", *domains, "--track2", "--splits", "train", "dev"],
               cwd=str(kit))
    elapsed = time.monotonic() - started
    report["baseline_seconds"] = round(elapsed, 1)
    print(f"\nofficial_baseline.py exited {proc.returncode} after {elapsed / 60:.1f} min")

    results: dict[str, dict] = {}
    for domain in domains:
        path = OUT_DIR / TRACK / domain / "results.json"
        if path.is_file():
            results[domain] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"  MISSING results for {domain}")
    if not results:
        die(f"official_baseline.py produced no results at all (exit {proc.returncode})")
    return results


def gate(results: dict[str, dict]) -> bool:
    """Per-domain comparison, then the equal-weight macro over domains (CLAUDE.md §4)."""
    print(f"\n  {'domain':<12}" + "".join(f"{k:>22}" for k in KEYS))
    print(f"  {'-' * 12}" + "".join(f"{'-' * 22:>22}" for _ in KEYS))

    passed = True
    rows: dict[str, dict] = {}
    for domain in sorted(results):
        cells, row = [], {}
        for key in KEYS:
            expected = PUBLISHED.get(domain, {}).get(key)
            observed = (results[domain].get(key) or {}).get("NDCG@10")
            counts = expected is not None and expected != 0.0
            ok = observed is not None and expected is not None and \
                abs(observed - expected) < TOLERANCE
            if counts and not ok:
                passed = False
            mark = "ok" if ok else ("--" if not counts else "XX")
            cells.append(f"{(observed if observed is not None else float('nan')):.4f}"
                         f"/{(expected if expected is not None else float('nan')):.4f} {mark:>3}")
            row[key] = {"published": expected, "observed": observed,
                        "counts": counts, "matched": ok}
        rows[domain] = row
        print(f"  {domain:<12}" + "".join(f"{c:>22}" for c in cells))

    # Macro: mean over domains, equal weight. Never a mean over topics.
    print(f"\n  {'key':<10} {'published':>10} {'observed':>10} {'delta':>10}  status")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}  ------")
    macro_rows = {}
    for key in KEYS:
        values = [(results[d].get(key) or {}).get("NDCG@10") for d in sorted(results)]
        values = [v for v in values if v is not None]
        observed = sum(values) / len(values) if values else None
        expected = PUBLISHED_MACRO[key]
        ok = observed is not None and abs(observed - expected) < TOLERANCE
        if not ok:
            passed = False
        macro_rows[key] = {"published": expected, "observed": observed,
                           "n_domains": len(values), "matched": ok}
        shown = f"{observed:10.4f}" if observed is not None else "     -    "
        delta = f"{observed - expected:+10.4f}" if observed is not None else "     -    "
        print(f"  {key:<10} {expected:10.4f} {shown} {delta}  "
              f"{'match' if ok else 'MISMATCH'}")

    report["gate_per_domain"] = rows
    report["gate_macro"] = macro_rows
    report["gate_passed"] = passed
    print(f"\n  Cells are observed/published. 'ok' matched to 4 dp, 'XX' diverged,")
    print( "  '--' is a published 0.0000 which abstains (a broken pipeline hits it too).")
    print( "  Macro is the equal-weight mean over domains, not over topics (§4).")
    return passed


# ============================================================================ --
def main() -> int:
    section(f"STAGE A — JDK (pyserini's Lucene jars need {MIN_JDK}+)")
    report["default_jdk"] = probe_java()
    found = find_jdk_home()
    if found is None:
        print(f"no installed JDK >= {MIN_JDK}; installing one")
        install_jdk()
        found = find_jdk_home()
        if found is None:
            die(f"no JDK >= {MIN_JDK} available after install; pyserini cannot run.")
    major, home = found
    activate_jdk(home)
    report["jdk_major"] = major
    print(f"JDK {major} OK")
    save_report()

    section(f"STAGE B — download all of {TRACK}")
    root = download()
    domains = list(report["domains"])  # type: ignore[arg-type]
    save_report()

    # Cheap and novel before expensive and confirmatory: a kernel killed in Stage D
    # still returns the audit.
    section("STAGE C — data audit (CLAUDE.md §9 Phase 2)")
    audit = run_audit(root, domains)
    report_guidance(audit)
    report["audit_file"] = str(AUDIT)
    save_report()
    print(f"\naudit written to {AUDIT}")

    section(f"STAGE D.1 — starter kit @ {KIT_COMMIT[:8]}")
    kit = clone_kit()
    save_report()

    section("STAGE D.2 — dependencies")
    install_deps()
    save_report()

    section("STAGE D.3 — official_baseline.py over 13 domains (this is the slow part)")
    results = run_baseline(kit, domains)
    report["results"] = results
    save_report()

    section("STAGE E — Phase 2 gate vs BASELINE_RESULTS.md @ 23093c3")
    passed = gate(results)
    save_report()

    section(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"report: {REPORT}")
    print(f"audit:  {AUDIT}")
    print(f"runs:   {OUT_DIR / TRACK}")
    if passed:
        print("\nPhase 2 gate closed. Pull the output, confirm locally with")
        print("  python eval/gate.py --all --results-dir cache/kernel_output/<slug>/baseline_out")
        print("then write notes/data_audit.md from data_audit.json.")
    else:
        print("\nGate not closed. The per-domain table above says which domain diverged.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
