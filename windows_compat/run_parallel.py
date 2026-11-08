"""Parallel batch driver: OFF -> C++ PSDR -> COMPOD.

Concurrency model
-----------------
* ``ThreadPoolExecutor(max_workers=--workers)``: one thread owns one model.
  A model pipeline consists of three *external processes*
  (off_to_pointcloud.py, psdr_cli.exe, run_psdr_compod.py).  The thread spends
  its time waiting on subprocess I/O with the GIL released, so W independent
  compute processes really execute simultaneously; the heavy Python/C++ code
  itself never shares a GIL.
* ``ProcessPoolExecutor(max_workers=--verify-procs)``: CPU-heavy result
  checking (mesh statistics, npz statistics, source-bbox comparison) runs in
  separate processes, outside the GIL.

Each model gets its own work directory and log.  ``summary.csv`` and
``progress.json`` are updated as models finish, so a long batch can be
interrupted and resumed (--skip-existing is on by default).

Usage:
  python windows_compat/run_parallel.py \
      --off-root data/oblique/vector_p_geometry_20261107 \
      --result-root data/oblique/result/vector_p_geometry_20261107 \
      --workers 6 --verify-procs 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "windows_compat", ".venv", "Scripts", "python.exe")
CLI = os.path.join(ROOT, "windows_compat", "psdr_cpp", "build", "psdr_cli.exe")
SAMPLE = os.path.join(ROOT, "windows_compat", "off_to_pointcloud.py")
COMPOD = os.path.join(ROOT, "windows_compat", "run_psdr_compod.py")
DEFAULT_OFF_ROOT = os.path.join(ROOT, "data", "oblique", "vector_p_geometry_20261107")
DEFAULT_RESULT_ROOT = os.path.join(ROOT, "data", "oblique", "result",
                                  "vector_p_geometry_20261107")
VCPKG_BIN = r"D:\github\vcpkg\installed\x64-windows\bin"
SDK_BIN = r"D:\Windows Kits\10\bin\10.0.26100.0\x64"

SUMMARY_FIELDS = [
    "dataset", "tag", "status", "seconds", "src_v", "src_f",
    "out_v", "out_f", "watertight", "planes", "inliers", "ply_bytes",
    "bbox_center_offset_m", "error", "verify_error", "log",
]


class _Tee:
    """Mirror stdout/stderr to a log file when launched without a shell."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def discover(off_root, datasets, only, limit):
    jobs = []
    for ds in datasets:
        d = os.path.join(off_root, ds)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".off"):
                continue
            tag = os.path.splitext(fn)[0]
            if only and tag not in only:
                continue
            jobs.append({"dataset": ds, "tag": tag, "off": os.path.join(d, fn)})
    if limit and limit > 0:
        jobs = jobs[:limit]
    return jobs


def verify_one(rec):
    """CPU-heavy post-processing, executed in a separate process."""
    out = {}
    if rec.get("status") not in ("ok", "skipped"):
        return out
    try:
        import numpy as np
        import trimesh

        m = trimesh.load(rec["ply"], process=False)
        v = np.asarray(m.vertices, dtype=np.float64)
        out["out_v"] = len(v)
        out["out_f"] = len(m.faces)
        out["watertight"] = bool(m.is_watertight)
        omin, omax = v.min(0), v.max(0)

        with np.load(rec["npz"]) as d:
            out["planes"] = int(np.asarray(d["group_parameters"]).shape[0])
            out["inliers"] = int(np.asarray(d["group_num_points"]).sum())

        # Stream the source OFF just for its bbox (no full mesh allocation).
        with open(rec["off"], "r", encoding="ascii", errors="replace") as fh:
            fh.readline()
            nv, nf, _ = [int(x) for x in fh.readline().split()[:3]]
            mn = np.full(3, np.inf)
            mx = np.full(3, -np.inf)
            for _ in range(nv):
                t = fh.readline().split()[:3]
                if len(t) < 3:
                    continue
                a = np.asarray([float(x) for x in t])
                np.minimum(mn, a, out=mn)
                np.maximum(mx, a, out=mx)
        out["src_v"] = nv
        out["src_f"] = nf
        out["bbox_center_offset_m"] = round(
            float(np.linalg.norm((mn + mx) / 2.0 - (omin + omax) / 2.0)), 3)
    except Exception as exc:
        out["verify_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _run_stage(cmd, log, timeout, stage, env):
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {stage} ===\n{' '.join(cmd)}\n")
        fh.flush()
        t = time.time()
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write(f"TIMEOUT after {timeout}s\n")
            raise
        fh.write(f"--- exit={p.returncode}  {time.time()-t:.1f}s ---\n")
    if p.returncode != 0:
        raise RuntimeError(f"{stage} exited {p.returncode}")


def run_one(job, cfg, ppool):
    """Drive one model's pipeline; called from a thread of the main process."""
    ds, tag = job["dataset"], job["tag"]
    work = os.path.join(cfg["result_root"], ds, tag)
    log_dir = os.path.join(cfg["result_root"], "_logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{ds}__{tag}.log")
    ply = os.path.join(work, "pointcloud.ply")
    center = os.path.join(work, "center.json")
    detected = os.path.join(work, "psdr_detected.npz")
    refined = os.path.join(work, "psdr_refined.npz")
    surf = os.path.join(work, f"{tag}_psdr_surface.ply")

    rec = {
        "dataset": ds, "tag": tag, "status": "pending", "seconds": 0.0,
        "off": job["off"], "ply": surf, "npz": refined, "log": log,
        "ply_bytes": 0, "error": "", "verify_error": "",
    }
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(f"model {ds}/{tag}\nsource {job['off']}\n")

    if cfg["skip_existing"] and os.path.isfile(surf) and os.path.getsize(surf) > 1000:
        rec["status"] = "skipped"
    else:
        t0 = time.time()
        try:
            _run_stage([PY, SAMPLE, job["off"], ply,
                        "--samples", str(cfg["samples"]),
                        "--center-json", center],
                       log, cfg["stage_timeout"], "1/3 pointcloud", cfg["env"])

            cli_cmd = [CLI, ply, "--out", refined, "--detected", detected,
                       "--epsilon-ratio", str(cfg["epsilon_ratio"]),
                       "--min-inliers", str(cfg["min_inliers"]),
                       "--knn", str(cfg["knn"]),
                       "--normal-th", str(cfg["normal_th"])]
            if cfg["max_seconds"] > 0:
                cli_cmd += ["--max-seconds", str(cfg["max_seconds"])]
            _run_stage(cli_cmd, log, cfg["stage_timeout"], "2/3 psdr", cfg["env"])

            _run_stage([PY, COMPOD, "--npz", refined,
                        "--center-json", center, "--out", surf],
                       log, cfg["stage_timeout"], "3/3 compod", cfg["env"])

            rec["status"] = "ok"
            rec["ply_bytes"] = os.path.getsize(surf)
        except subprocess.TimeoutExpired as exc:
            rec["status"] = "timeout"
            rec["error"] = f"stage timeout {cfg['stage_timeout']}s: {exc.cmd[0]}"
        except Exception as exc:
            rec["status"] = "fail"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = round(time.time() - t0, 1)

    if ppool is not None:
        try:
            rec.update(ppool.submit(verify_one, rec).result(timeout=600))
        except Exception as exc:
            rec["verify_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off-root", default=DEFAULT_OFF_ROOT)
    ap.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    ap.add_argument("--dataset", action="append", default=None,
                    help="repeatable; default = anhui + moganshan")
    ap.add_argument("--only", action="append", default=[],
                    help="repeatable model tag, e.g. smid_004")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent model pipelines (threads + external processes)")
    ap.add_argument("--verify-procs", type=int, default=2,
                    help="processes for CPU-heavy result verification")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--epsilon-ratio", type=float, default=0.01)
    ap.add_argument("--min-inliers", type=int, default=120)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--normal-th", type=float, default=0.9)
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="PSDR refine budget, 0 = converge")
    ap.add_argument("--stage-timeout", type=int, default=1800)
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false", default=True)
    ap.add_argument("--log-file", default="",
                    help="mirror progress here (for detached launches)")
    a = ap.parse_args()

    if a.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(a.log_file)), exist_ok=True)
        _log_fh = open(a.log_file, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)
        print(f"\n=== batch start {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
              flush=True)

    for label, path in [("python", PY), ("psdr_cli", CLI),
                        ("sampler", SAMPLE), ("compod", COMPOD)]:
        if not os.path.isfile(path):
            print(f"missing {label}: {path}", file=sys.stderr)
            return 2

    datasets = a.dataset or ["anhui", "moganshan"]
    jobs = discover(a.off_root, datasets, set(a.only) if a.only else None, a.limit)
    if not jobs:
        print("no OFF files matched", file=sys.stderr)
        return 2
    os.makedirs(a.result_root, exist_ok=True)
    summary = os.path.join(a.result_root, "summary.csv")
    progress = os.path.join(a.result_root, "progress.json")

    with open(summary, "w", encoding="utf-8-sig", newline="") as fh:
        csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS).writeheader()

    env = os.environ.copy()
    env["PATH"] = VCPKG_BIN + os.pathsep + SDK_BIN + os.pathsep + env.get("PATH", "")
    cfg = {
        "result_root": a.result_root, "samples": a.samples,
        "epsilon_ratio": a.epsilon_ratio, "min_inliers": a.min_inliers,
        "knn": a.knn, "normal_th": a.normal_th,
        "max_seconds": a.max_seconds, "stage_timeout": a.stage_timeout,
        "skip_existing": a.skip_existing, "env": env,
    }

    print(f"models={len(jobs)}  workers(threads)={a.workers}  "
          f"verify_procs={a.verify_procs}  skip_existing={a.skip_existing}",
          flush=True)
    print(f"result : {a.result_root}", flush=True)
    lock = threading.Lock()
    counts = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=a.verify_procs) as ppool, \
         ThreadPoolExecutor(max_workers=a.workers) as tpool:
        futures = [tpool.submit(run_one, j, cfg, ppool) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"dataset": "?", "tag": "?", "status": "fail",
                       "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                with open(summary, "a", encoding="utf-8-sig", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS,
                                   extrasaction="ignore").writerow(rec)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                with open(progress, "w", encoding="utf-8") as fh:
                    json.dump({
                        "total": len(jobs), "done": i, "counts": counts,
                        "elapsed_s": round(time.time() - t0, 1),
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "result_root": a.result_root,
                    }, fh, indent=2, ensure_ascii=False)
            print(f"[{i}/{len(jobs)}] {rec['dataset']}/{rec['tag']} "
                  f"{rec['status']} {rec.get('seconds', 0)}s "
                  f"out_v={rec.get('out_v', '-')} planes={rec.get('planes', '-')} "
                  f"offset={rec.get('bbox_center_offset_m', '-')} "
                  f"{rec.get('error', '')}{rec.get('verify_error', '')}",
                  flush=True)

    bad = counts.get("fail", 0) + counts.get("timeout", 0)
    print("-" * 60)
    print(f"done={sum(counts.values())}  {counts}  {time.time()-t0:.1f}s")
    print(f"summary : {summary}")
    print(f"progress: {progress}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
