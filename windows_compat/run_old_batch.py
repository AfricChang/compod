#!/usr/bin/env python
"""Chained follow-up batch for the older ``margin_4m`` OFF dataset.

Waits until the currently running ``vector_p_geometry_20261107`` batch *and*
its PLY->OFF conversion have finished, then:

1. runs ``run_parallel.py`` on ``data/oblique/{anhui,moganshan}`` (49 + 116
   = 165 OFF files) into an independent result tree;
2. runs ``ply2off.py`` on those PLYs into an independent OFF tree.

Everything uses absolute paths and ``--skip-existing``-style resumption, so
the chain can be interrupted and restarted safely.  All output stays under
``data/oblique/`` and never touches the existing vector_p_geometry results.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "windows_compat" / ".venv" / "Scripts" / "python.exe"
RUN_PARALLEL = ROOT / "windows_compat" / "run_parallel.py"
PLY2OFF = ROOT / "windows_compat" / "ply2off.py"

NEW_RESULT = ROOT / "data" / "oblique" / "result" / "vector_p_geometry_20261107"
NEW_OFF = ROOT / "data" / "oblique" / "off_recon"
WATCH_LOG = ROOT / "windows_compat" / "ply2off_watch.log"

OLD_OFF_ROOT = ROOT / "data" / "oblique"
OLD_RESULT = ROOT / "data" / "oblique" / "result" / "margin_4m"
OLD_OUT = ROOT / "data" / "oblique" / "off_recon_margin_4m"
CHAIN_LOG = ROOT / "windows_compat" / "old_batch_chain.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def progress_done(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return int(d.get("done", 0)), int(d.get("total", 0))
    except Exception:
        return 0, 0


def off_count(root: Path):
    return sum(1 for ds in ("anhui", "moganshan")
               for _ in (root / ds).glob("*.off")) if root.is_dir() else 0


def run_stage(name, cmd, log_path: Path):
    log(f"{name}: start")
    log(f"{name}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                           stdout=fh, stderr=subprocess.STDOUT)
    log(f"{name}: exit={p.returncode} in {time.time()-t0:.0f}s  log={log_path}")
    return p.returncode


def wait_for_current_batch(timeout_s):
    """Wait for the vector batch, then for its OFF watcher to report [final]."""
    t0 = time.time()
    last_done, last_change = -1, t0
    log("stage 0: waiting for vector_p_geometry_20261107 batch to finish")
    while True:
        done, total = progress_done(NEW_RESULT / "progress.json")
        if done != last_done:
            last_done, last_change = done, time.time()
        if total and done >= total:
            log(f"stage 0: batch finished {done}/{total}")
            break
        if time.time() - t0 > timeout_s:
            log(f"stage 0: TIMEOUT at {done}/{total}, continuing anyway")
            break
        # If the driving session died, don't stall for the full timeout.
        if time.time() - last_change > 600:
            log(f"stage 0: no progress for 10min at {done}/{total}, "
                f"assuming the batch stopped; continuing")
            break
        if int(time.time() - t0) % 300 < 20:
            log(f"stage 0: batch {done}/{total}")
        time.sleep(20)

    log("stage 0b: waiting for PLY->OFF watcher to finish")
    t1 = time.time()
    while time.time() - t1 < 1200:
        try:
            if "[final]" in WATCH_LOG.read_text(encoding="utf-8", errors="replace"):
                log("stage 0b: watcher reported [final]")
                break
        except OSError:
            pass
        time.sleep(20)
    else:
        log("stage 0b: watcher did not report [final] in 20min; "
            "running one explicit final conversion pass")
        run_stage("stage 0c final vector PLY->OFF",
                  [PY, "-u", PLY2OFF,
                   "--result-root", NEW_RESULT,
                   "--src-root", ROOT / "data" / "oblique" / "vector_p_geometry_20261107",
                   "--out", NEW_OFF],
                  ROOT / "windows_compat" / "old_batch_stage0c.log")
    log(f"stage 0 done: vector OFF count = {off_count(NEW_OFF)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true",
                    help="skip waiting for the current batch (manual restart)")
    ap.add_argument("--wait-timeout", type=float, default=5400.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-procs", type=int, default=2)
    a = ap.parse_args()

    log("=" * 70)
    log(f"chain start  old_off_root={OLD_OFF_ROOT}")
    log(f"  old PLY result : {OLD_RESULT}")
    log(f"  old OFF output : {OLD_OUT}")

    if not a.no_wait:
        wait_for_current_batch(a.wait_timeout)

    done, total = progress_done(OLD_RESULT / "progress.json")
    if total and done >= total:
        log(f"stage 1: old batch already complete ({done}/{total}), skipping")
        rc1 = 0
    else:
        rc1 = run_stage(
            "stage 1 old OFF -> PSDR -> COMPOD PLY",
            [PY, "-u", RUN_PARALLEL,
             "--off-root", OLD_OFF_ROOT,
             "--result-root", OLD_RESULT,
             "--workers", a.workers,
             "--verify-procs", a.verify_procs,
             "--stage-timeout", 1800,
             "--log-file", ROOT / "windows_compat" / "old_batch.log"],
            ROOT / "windows_compat" / "old_batch.console.log")

    rc2 = run_stage(
        "stage 2 old PLY -> n-gon OFF",
        [PY, "-u", PLY2OFF,
         "--result-root", OLD_RESULT,
         "--src-root", OLD_OFF_ROOT,
         "--out", OLD_OUT],
        ROOT / "windows_compat" / "old_batch_ply2off.log")

    d, t = progress_done(OLD_RESULT / "progress.json")
    report = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_result_root": str(OLD_RESULT),
        "old_off_root": str(OLD_OUT),
        "batch_done": d, "batch_total": t,
        "batch_exit": rc1, "ply2off_exit": rc2,
        "off_files": off_count(OLD_OUT),
        "vector_off_files": off_count(NEW_OFF),
    }
    (ROOT / "windows_compat" / "old_batch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log("stage 3: " + json.dumps(report, ensure_ascii=False))
    log("chain done")
    return 0 if rc1 == 0 and rc2 == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
