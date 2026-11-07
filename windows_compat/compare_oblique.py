"""Geometric A/B comparison of reconstructed PLYs against a source OFF.

For each PLY we sample both surfaces and report a symmetric point-to-surface
Chamfer-style distance (nearest-neighbour on densely sampled triangles), plus
mesh complexity and watertightness.  Units are the OFF's own units (metres for
this dataset).

Usage:
  python windows_compat/compare_oblique.py \
      --off data/oblique/anhui/smid_004.off \
      --ply "PSDR refined=data/oblique/result/psdr_cpp/sm4_refined/smid_004_psdr_surface.ply" \
      --ply "PSDR detect=data/oblique/result/psdr_cpp/sm4_norefine/smid_004_psdr_surface.ply"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def read_off(path):
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        fh.readline()
        nv, nf, _ = map(int, fh.readline().split()[:3])
        v = np.array([[float(x) for x in fh.readline().split()[:3]]
                      for _ in range(nv)])
        f = np.array([[int(x) for x in fh.readline().split()[1:4]]
                      for _ in range(nf)])
    return v, f


def surface_samples(verts, faces, n, rng):
    tri = verts[faces]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1) / 2.0
    area = np.maximum(area, 1e-12)
    ids = rng.choice(len(faces), size=n, p=area / area.sum())
    u = np.sqrt(rng.random(n))[:, None]
    v = rng.random(n)[:, None]
    return (1 - u) * tri[ids, 0] + u * (1 - v) * tri[ids, 1] + u * v * tri[ids, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--ply", action="append", required=True,
                    help="label=path (repeatable)")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--per-face", type=int, default=8)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    V, F = read_off(a.off)
    ref = surface_samples(V, F, a.samples, rng)
    tree_ref = cKDTree(ref)
    probe = ref[rng.choice(len(ref), min(6000, len(ref)), replace=False)]

    print(f"OFF: {os.path.basename(a.off)}  V={len(V):,} F={len(F):,}  "
          f"bbox={np.round(V.min(0), 1).tolist()} .. "
          f"{np.round(V.max(0), 1).tolist()}")
    hdr = ("%-26s %7s %7s %11s %9s %9s %9s %9s" %
           ("PLY", "V", "F", "watertight", "mean", "median", "p90", "sec"))
    print(hdr)
    print("-" * len(hdr))
    import time
    for spec in a.ply:
        if "=" not in spec:
            print("bad --ply spec (need label=path):", spec)
            continue
        label, path = spec.split("=", 1)
        t0 = time.time()
        m = trimesh.load(path, process=False)
        P = surface_samples(m.vertices, m.faces,
                            max(2000, len(m.faces) * a.per_face), rng)
        d1 = cKDTree(P).query(probe)[0]
        d2 = tree_ref.query(P)[0]
        d = np.concatenate([d1, d2])
        print("%-26s %7d %7d %11s %9.3f %9.3f %9.3f %9.1f" %
              (label[:26], len(m.vertices), len(m.faces),
               m.is_watertight, d.mean(), np.median(d),
               np.percentile(d, 90), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
