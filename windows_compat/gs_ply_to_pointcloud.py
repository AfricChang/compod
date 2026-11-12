"""3D Gaussian-Splatting (or any scalar-attribute) PLY -> PSDR-ready point cloud.

The PSDR C++ front end reads a binary little-endian PLY with x/y/z and
nx/ny/nz.  3DGS PLYs usually carry dozens of extra float attributes and often
have *zero* normals (the fields exist but are all 0).  PSDR's own fallback
(jet_estimate_normals) produces *unoriented* normals, which is fine for region
growing but breaks COMPOD's `label_partition(mode="normals")`: that step votes
cell inside/outside from `dot(normal, centroid - point)` and therefore needs
normals that point away from the solid.

Two modes:

  * **full / lossless** (recommended when the source must be preserved):
    `--samples 0 --no-filter --no-center`.  Every input point is kept, in the
    original coordinate frame, with only the missing normals filled in by
    chunked local-PCA.  No alpha/scale filtering, no voxel balancing, no
    subsampling, no translation.

  * **reduced** (fast iteration): filter by opacity/scale and subsample.

Normals are estimated in chunks so a multi-million-point cloud never materialises
the full (N, k, 3) neighbour tensor at once.  Orientation is fixed per point
towards/away from the bbox centre; `--consistent` additionally runs Open3D's
MST-based propagation (only practical below ~1M points).

Usage:
  python windows_compat/gs_ply_to_pointcloud.py <in.ply> <out.ply> \
      --center-json <center.json> --samples 0 --no-filter --no-center \
      [--knn 24] [--orient outward|inward|none] [--stats <json>]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

TYPE_MAP = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}


def read_header(path):
    lines = []
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("PLY has no end_header")
            line = raw.decode("ascii", "replace").strip()
            lines.append(line)
            if line == "end_header":
                return lines, fh.tell()
    raise AssertionError("unreachable")


def parse_ply(path):
    lines, offset = read_header(path)
    fmt = next((l for l in lines if l.startswith("format ")), "")
    if "binary_little_endian" not in fmt:
        raise ValueError("only binary_little_endian PLY is supported: " + fmt)
    elements = []
    cur = None
    for line in lines:
        if line.startswith("element "):
            parts = line.split()
            cur = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(cur)
        elif line.startswith("property ") and cur is not None:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("list properties are not supported")
            cur["props"].append((parts[-1], parts[1]))
    if len(elements) != 1 or elements[0]["name"] != "vertex":
        raise ValueError("expected a single `vertex` element, got: %s" %
                         [(e["name"], e["count"]) for e in elements])
    props = elements[0]["props"]
    for _, t in props:
        if t not in TYPE_MAP:
            raise ValueError("unsupported PLY property type: " + t)
    dt = np.dtype([(n, TYPE_MAP[t]) for n, t in props])
    mm = np.memmap(path, dtype=dt, mode="r", offset=offset,
                   shape=(elements[0]["count"],))
    return mm, offset, elements[0]["count"], len(props)


def collect_xyz(mm, count, alpha_th, max_scale, no_filter, chunk=400_000):
    """Read x/y/z for every point; optionally drop low-opacity/large Gaussians."""
    fields = mm.dtype.names
    has_alpha = "opacity" in fields
    scale_cols = [c for c in fields if c.startswith("scale_")]
    kept = []
    dropped_nonfinite = 0
    n_kept = 0
    for start in range(0, count, chunk):
        blk = mm[start:start + chunk]
        xyz = np.column_stack([blk[c] for c in ("x", "y", "z")]).astype(np.float64)
        mask = np.isfinite(xyz).all(axis=1)
        dropped_nonfinite += int((~mask).sum())
        if not no_filter:
            if has_alpha:
                logit = np.asarray(blk["opacity"], dtype=np.float64)
                alpha = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
                mask &= alpha >= alpha_th
            if scale_cols and max_scale > 0:
                sc = np.column_stack([blk[c] for c in scale_cols]).astype(np.float64)
                mask &= np.exp(np.clip(sc, -30.0, 10.0)).max(axis=1) <= max_scale
        kept.append(xyz[mask].astype(np.float32))
        n_kept += int(mask.sum())
        if count > 2_000_000 and start % 2_000_000 == 0:
            print("  read %d / %d vertices" % (start + len(blk), count), flush=True)
    return np.concatenate(kept), n_kept, dropped_nonfinite


def subsample(pts, n_samples, voxel, rng):
    n = len(pts)
    if n_samples <= 0 or n_samples >= n:
        return np.arange(n, dtype=np.int64)
    order = rng.permutation(n)
    if voxel > 0:
        vi = np.floor(pts / voxel).astype(np.int64)
        vi -= vi.min(axis=0)
        dims = vi.max(axis=0) + 2
        key = (vi[:, 0] * dims[1] * dims[2]
               + vi[:, 1] * dims[2] + vi[:, 2])
        _, first = np.unique(key[order], return_index=True)
        chosen = order[first]
        rng.shuffle(chosen)
        if len(chosen) >= n_samples:
            return chosen[:n_samples]
        pool = np.setdiff1d(order, chosen, assume_unique=False)
        extra = rng.choice(pool, size=n_samples - len(chosen),
                           replace=len(pool) < n_samples - len(chosen))
        return np.concatenate([chosen, extra]).astype(np.int64)
    return order[:n_samples].astype(np.int64)


def estimate_normals(pts, knn, chunk=100_000, workers=-1):
    """Chunked local-plane PCA normals; returns float64 (N, 3), unit length."""
    from scipy.spatial import cKDTree
    print("building KD-tree on %d points ..." % len(pts), flush=True)
    tree = cKDTree(pts)
    nrm = np.empty_like(pts)
    t0 = time.time()
    for start in range(0, len(pts), chunk):
        q = pts[start:start + chunk]
        _, nb = tree.query(q, k=knn + 1, workers=workers)
        diff = (pts[nb[:, 1:]] - q[:, None, :]).astype(np.float32)
        cov = np.einsum("nki,nkj->nij", diff, diff) / max(knn - 1, 1)
        _, vecs = np.linalg.eigh(cov)      # ascending eigenvalues
        n = vecs[:, :, 0].astype(np.float64)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        nrm[start:start + chunk] = n
        if len(pts) > 500_000:
            print("  normals %d / %d  (%.1fs)" %
                  (min(start + chunk, len(pts)), len(pts), time.time() - t0),
                  flush=True)
    return nrm


def consistent_orientation(pts, nrm):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm.copy())
    pcd.orient_normals_consistent_tangent_plane(10)
    out = np.asarray(pcd.normals).copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_in")
    ap.add_argument("ply_out")
    ap.add_argument("--center-json", required=True)
    ap.add_argument("--samples", type=int, default=200_000,
                    help="0 = keep every point (no subsampling)")
    ap.add_argument("--no-filter", action="store_true",
                    help="keep every Gaussian regardless of opacity/scale")
    ap.add_argument("--no-center", action="store_true",
                    help="keep original coordinates (center.json gets [0,0,0])")
    ap.add_argument("--voxel", type=float, default=0.0)
    ap.add_argument("--opacity-th", type=float, default=0.5)
    ap.add_argument("--max-scale", type=float, default=0.05)
    ap.add_argument("--knn", type=int, default=24)
    ap.add_argument("--normal-chunk", type=int, default=100_000)
    ap.add_argument("--orient", choices=("outward", "inward", "none"),
                    default="outward")
    ap.add_argument("--consistent", action="store_true",
                    help="MST-based normal propagation (skipped above 1M points)")
    ap.add_argument("--zero-normals", action="store_true",
                    help="write zero normals and let PSDR's jet estimator run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()

    t0 = time.time()
    mm, offset, count, n_props = parse_ply(a.ply_in)
    fields = mm.dtype.names
    print("input: %s  vertices=%d  properties=%d  header=%dB" %
          (a.ply_in, count, n_props, offset), flush=True)

    zero_normals = True
    if all(c in fields for c in ("nx", "ny", "nz")):
        probe = np.column_stack([mm[c][:50_000] for c in ("nx", "ny", "nz")])
        zero_normals = not np.any(np.linalg.norm(probe, axis=1) > 1e-8)
    print("input normals: %s" % ("all zero / absent" if zero_normals
                                else "present"), flush=True)

    pts_all, n_kept, n_nonfinite = collect_xyz(
        mm, count, a.opacity_th, a.max_scale, a.no_filter)
    print("points kept: %d / %d (filter=%s, non-finite dropped=%d)" %
          (n_kept, count, "off" if a.no_filter else "on", n_nonfinite),
          flush=True)
    if n_kept == 0:
        raise ValueError("no points left; relax --opacity-th / --max-scale")

    rng = np.random.default_rng(a.seed)
    sel = subsample(pts_all, a.samples, a.voxel, rng)
    pts = np.ascontiguousarray(pts_all[sel], dtype=np.float64)
    del pts_all

    if a.zero_normals:
        # Diagnostics only: PSDR's jet normals are unoriented, so COMPOD's
        # normal-based cell labelling will not be meaningful.
        nrm = np.zeros_like(pts)
        normal_method = "zero (PSDR jet fallback, unoriented)"
    else:
        nrm = estimate_normals(pts, a.knn, a.normal_chunk)
        normal_method = "PCA kNN=%d" % a.knn
        if a.consistent and len(pts) <= 1_000_000:
            nrm = consistent_orientation(pts, nrm)
            normal_method += " + consistent"
        elif a.consistent:
            print("  --consistent skipped (too many points)", flush=True)
        if a.orient != "none":
            centre0 = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            dp = np.einsum("ij,ij->i", nrm, pts - centre0)
            flip = dp < 0 if a.orient == "outward" else dp > 0
            nrm[flip] *= -1.0
            print("orientation: %s (per-point flipped %d / %d)" %
                  (a.orient, int(flip.sum()), len(pts)), flush=True)
            normal_method += " + " + a.orient
        normal_method += " (per-point)"

    centre = (np.zeros(3) if a.no_center
              else (pts.min(axis=0) + pts.max(axis=0)) / 2.0)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    data = np.hstack([pts - centre, nrm]).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(a.ply_out)), exist_ok=True)
    with open(a.ply_out, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n"
                  "element vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property float nx\nproperty float ny\nproperty float nz\n"
                  "end_header\n" % len(pts)).encode("ascii"))
        data.tofile(fh)

    os.makedirs(os.path.dirname(os.path.abspath(a.center_json)), exist_ok=True)
    meta = {"center": centre.tolist(), "bbox_diag": diag,
            "source": os.path.basename(a.ply_in), "points": int(len(pts)),
            "normals": normal_method, "seed": a.seed,
            "filtered": not a.no_filter, "subsampled": a.samples > 0}
    with open(a.center_json, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    stats = {**meta, "input_points": int(count),
             "points_kept_before_sampling": int(n_kept),
             "nonfinite_dropped": int(n_nonfinite),
             "seconds": round(time.time() - t0, 1),
             "ply_out": os.path.abspath(a.ply_out),
             "ply_bytes": os.path.getsize(a.ply_out)}
    if a.stats:
        with open(a.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
    print("pointcloud: %s" % json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
