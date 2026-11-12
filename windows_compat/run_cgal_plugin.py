"""CGALLAB plugin runner: bridge between CGALLAB C++ plugin and PSDR+COMPOD.

Usage:
    python windows_compat/run_cgal_plugin.py \
        --mode all|psdr|compod \
        --input <pointcloud.ply> \
        --out-dir <output_directory> \
        [--center-json <center.json>] \
        [--epsilon-ratio 0.01] [--min-inliers 120] [--knn 10] [--normal-th 0.9] \
        [--no-refine] [--max-seconds -1] \
        [--reg-area 0.8] [--no-simplify] [--save-cells]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "windows_compat"))
os.chdir(ROOT)

from ply_coords import shift_ascii_ply  # noqa: E402
from pycompod import PolyhedralComplex, VertexGroup  # noqa: E402
from run_psdr_compod import sanitize  # noqa: E402


def compute_ply_center(path: str) -> list[float]:
    """Compute bounding box center of an ASCII or binary PLY."""
    min_c = [float("inf")] * 3
    max_c = [float("-inf")] * 3
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        header = []
        for line in fh:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        fmt = "ascii"
        n_verts = 0
        for h in header:
            if h.startswith("format binary"):
                fmt = "binary"
            elif h.startswith("element vertex"):
                n_verts = int(h.split()[-1])

        if fmt == "ascii":
            for _ in range(n_verts):
                parts = fh.readline().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    min_c[0] = min(min_c[0], x)
                    min_c[1] = min(min_c[1], y)
                    min_c[2] = min(min_c[2], z)
                    max_c[0] = max(max_c[0], x)
                    max_c[1] = max(max_c[1], y)
                    max_c[2] = max(max_c[2], z)
    if min_c[0] == float("inf"):
        return [0.0, 0.0, 0.0]
    return [(min_c[i] + max_c[i]) / 2.0 for i in range(3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "psdr", "compod"], default="all")
    ap.add_argument("--input", default="")
    ap.add_argument("--npz", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="model")
    ap.add_argument("--center-json", default="")

    # PSDR parameters
    ap.add_argument("--epsilon-ratio", type=float, default=0.01)
    ap.add_argument("--min-inliers", type=int, default=120)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--normal-th", type=float, default=0.9)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=-1)
    ap.add_argument("--max-iter", type=int, default=-1)

    # COMPOD parameters
    ap.add_argument("--reg-area", type=float, default=0.8)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--save-cells", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = args.stem

    # Resolve center offset
    center = [0.0, 0.0, 0.0]
    if args.center_json and os.path.isfile(args.center_json):
        with open(args.center_json, "r", encoding="utf-8") as fh:
            center = json.load(fh).get("center", [0.0, 0.0, 0.0])
    else:
        cj_path = os.path.join(args.out_dir, f"{stem}_center.json")
        if os.path.isfile(cj_path):
            with open(cj_path, "r", encoding="utf-8") as fh:
                center = json.load(fh).get("center", [0.0, 0.0, 0.0])

    psdr_cli_exe = os.path.join(ROOT, "windows_compat", "psdr_cpp", "build", "psdr_cli.exe")
    if not os.path.isfile(psdr_cli_exe):
        print(f"[ERROR] psdr_cli.exe not found at {psdr_cli_exe}", flush=True)
        return 1

    psdr_npz = args.npz or os.path.join(args.out_dir, f"{stem}_psdr.npz")
    psdr_ply = os.path.join(args.out_dir, f"{stem}_psdr.ply")
    compod_ply = os.path.join(args.out_dir, f"{stem}_compod.ply")

    res: dict = {
        "status": "pending",
        "mode": args.mode,
        "stem": stem,
        "psdr_ply": "",
        "psdr_npz": "",
        "compod_ply": "",
        "planes": 0,
        "cells": 0,
    }

    # Step 1: Run PSDR if needed
    if args.mode in ("all", "psdr"):
        if not args.input or not os.path.isfile(args.input):
            print(f"[ERROR] Valid input point cloud PLY required for PSDR: {args.input}", flush=True)
            return 1

        cmd = [
            psdr_cli_exe,
            args.input,
            "--out", psdr_npz,
            "--ply", psdr_ply,
            "--epsilon-ratio", str(args.epsilon_ratio),
            "--min-inliers", str(args.min_inliers),
            "--knn", str(args.knn),
            "--normal-th", str(args.normal_th),
        ]
        if args.no_refine:
            cmd.append("--no-refine")
        if args.max_seconds > 0:
            cmd.extend(["--max-seconds", str(args.max_seconds)])
        if args.max_iter > 0:
            cmd.extend(["--max-iter", str(args.max_iter)])

        print(f"[PROGRESS 10%] Running PSDR: {' '.join(cmd)}", flush=True)
        t_psdr0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(p.stdout.readline, ""):
            print(f"[PSDR] {line.strip()}", flush=True)
        p.stdout.close()
        rc = p.wait()
        if rc != 0:
            print(f"[ERROR] psdr_cli.exe failed with exit code {rc}", flush=True)
            return rc
        print(f"[PROGRESS 50%] PSDR completed in {time.time() - t_psdr0:.2f}s", flush=True)

        if os.path.isfile(psdr_ply):
            shift_ascii_ply(psdr_ply, center)
            res["psdr_ply"] = psdr_ply
        res["psdr_npz"] = psdr_npz

    # Step 2: Run COMPOD if needed
    if args.mode in ("all", "compod"):
        if not os.path.isfile(psdr_npz):
            print(f"[ERROR] PSDR .npz file not found for COMPOD: {psdr_npz}", flush=True)
            return 1

        print("[PROGRESS 60%] Initializing COMPOD...", flush=True)
        t_compod0 = time.time()
        sanitized_npz, stats = sanitize(psdr_npz, keep_classes=False)
        res["planes"] = stats.get("planes", 0)

        verbosity = logging.DEBUG if args.verbose else logging.WARNING
        vg = VertexGroup(sanitized_npz, verbosity=verbosity, debug_export=False)
        cc = PolyhedralComplex(vg, device="cpu", verbosity=verbosity)

        print("[PROGRESS 70%] COMPOD: Constructing plane partition...", flush=True)
        cc.construct_partition()
        cc.add_bounding_box_planes()

        print("[PROGRESS 80%] COMPOD: Labelling partition (graph cuts)...", flush=True)
        cc.label_partition(mode="normals", regularization={"area": args.reg_area})

        if not args.no_simplify:
            print("[PROGRESS 90%] COMPOD: Simplifying partition...", flush=True)
            cc.simplify_partition_tree_based()

        print("[PROGRESS 95%] COMPOD: Saving reconstructed surface...", flush=True)
        cc.save_surface(out_file=compod_ply, triangulate=False)
        shift_ascii_ply(compod_ply, center)
        res["compod_ply"] = compod_ply

        if args.save_cells:
            cells_ply = os.path.join(args.out_dir, f"{stem}_compod_cells.ply")
            cc.save_in_cells(out_file=cells_ply)
            shift_ascii_ply(cells_ply, center)
            res["cells_ply"] = cells_ply

        cell_count = len(cc.cells) if getattr(cc, "cells", None) is not None else 0
        res["cells"] = cell_count
        print(f"[PROGRESS 100%] COMPOD finished: {cell_count} cells in {time.time() - t_compod0:.2f}s", flush=True)

    res["status"] = "ok"
    print("JSON_RESULT: " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
