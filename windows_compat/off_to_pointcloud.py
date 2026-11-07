"""OFF -> PSDR point cloud (binary little-endian PLY with normals).

Same sampling as the official pipeline's sample_points.py (area-weighted
surface samples, true face normals), but the coordinates are centred on the
OFF bbox centre.  Large geodetic coordinates would blow up COMPOD's exact
rational kernel, so the centre is written to a JSON sidecar and
run_psdr_compod.py shifts the reconstructed surface back.

Usage:
  python windows_compat/off_to_pointcloud.py <in.off> <out.ply> \
      [--samples 200000] --center-json <center.json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_off(path):
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        if fh.readline().strip() != "OFF":
            raise ValueError("not an ASCII OFF: " + path)
        nv, nf, _ = [int(x) for x in fh.readline().split()[:3]]
        v = np.empty((nv, 3), dtype=np.float64)
        for i in range(nv):
            v[i] = [float(x) for x in fh.readline().split()[:3]]
        f = np.empty((nf, 3), dtype=np.int64)
        for i in range(nf):
            t = fh.readline().split()
            f[i] = [int(t[1]), int(t[2]), int(t[3])]
    return v, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("ply")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--center-json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    v, f = load_off(a.off)
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cr, axis=1)
    area = np.maximum(length / 2.0, 1e-12)
    fn = cr / np.maximum(length, 1e-30)[:, None]

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(f), size=a.samples, p=area / area.sum())
    u = np.sqrt(rng.random(a.samples))
    w = rng.random(a.samples)
    p = ((1.0 - u)[:, None] * tri[sel, 0]
         + (u * (1.0 - w))[:, None] * tri[sel, 1]
         + u[:, None] * w[:, None] * tri[sel, 2])

    center = (v.min(0) + v.max(0)) / 2.0
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    p -= center

    data = np.hstack([p, fn[sel]]).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.ply)), exist_ok=True)
    with open(a.ply, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  "element vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float nx\nproperty float ny\nproperty float nz\n"
                  "end_header\n" % a.samples).encode("ascii"))
        data.tofile(fh)

    os.makedirs(os.path.dirname(os.path.abspath(a.center_json)), exist_ok=True)
    with open(a.center_json, "w", encoding="utf-8") as fh:
        json.dump({"center": center.tolist(), "bbox_diag": diag,
                   "off": os.path.basename(a.off), "points": a.samples},
                  fh, indent=2)
    print("pointcloud: %s  points=%d  bbox_diag=%.3f  center=%s" %
          (a.ply, a.samples, diag, np.round(center, 3).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
