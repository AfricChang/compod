"""Process B: feed an official PSDR .npz straight into COMPOD.

    python windows_compat/run_psdr_compod.py \
        --npz work/psdr_refined.npz \
        --center-json work/center.json \
        --out result/sm4_psdr_surface.ply

COMPOD's VertexGroup reads: points, normals, group_parameters, group_colors,
group_num_points, group_points (int32), classes.  PSDR writes all of them, but
its `classes` array is point-wise; a non-empty classes array changes the
labelling path, so a sanitised copy with an empty `classes` is used by default
(--keep-classes disables this).  The reconstructed surface is shifted back
into the source OFF frame using the centre recorded by off_to_pointcloud.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "windows_compat"))  # sage.all shim
os.chdir(ROOT)

from ply_coords import shift_ascii_ply  # noqa: E402
from pycompod import PolyhedralComplex, VertexGroup  # noqa: E402


def sanitize(npz_path: str, keep_classes: bool) -> tuple[str, dict]:
    data = dict(np.load(npz_path))
    stats = {
        "points": int(np.asarray(data["points"]).shape[0]),
        "planes": int(np.asarray(data["group_parameters"]).shape[0]),
        "group_points_dtype": str(np.asarray(data["group_points"]).dtype),
        "classes_len": int(np.asarray(data.get("classes", [])).size),
    }
    if keep_classes:
        return npz_path, stats
    cls = data.get("classes")
    if cls is None or np.asarray(cls).size == 0:
        return npz_path, stats
    out = os.path.splitext(npz_path)[0] + "_compod.npz"
    data["classes"] = np.zeros(0, dtype=np.int32)
    np.savez(out, **data)
    stats["sanitized"] = out
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--center-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-classes", action="store_true")
    ap.add_argument("--reg-area", type=float, default=0.8)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--save-cells", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    with open(a.center_json, "r", encoding="utf-8") as fh:
        center = json.load(fh)["center"]

    t0 = time.time()
    npz_path, stats = sanitize(a.npz, a.keep_classes)
    verbosity = logging.DEBUG if a.verbose else logging.WARNING
    vg = VertexGroup(npz_path, verbosity=verbosity, debug_export=False)
    cc = PolyhedralComplex(vg, device="cpu", verbosity=verbosity)
    cc.construct_partition()
    cc.add_bounding_box_planes()
    cc.label_partition(mode="normals", regularization={"area": a.reg_area})
    if not a.no_simplify:
        cc.simplify_partition_tree_based()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.save_cells:
        cells = os.path.join(os.path.dirname(a.out), "volume_mesh.ply")
        cc.save_in_cells(out_file=cells)
        shift_ascii_ply(cells, center)
    cc.save_surface(out_file=a.out, triangulate=False)
    shift_ascii_ply(a.out, center)

    print("PSDR->COMPOD %s" % json.dumps({
        **stats,
        "cells": len(cc.cells) if getattr(cc, "cells", None) is not None else -1,
        "seconds": round(time.time() - t0, 1),
        "ply_bytes": os.path.getsize(a.out),
        "output": a.out,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
