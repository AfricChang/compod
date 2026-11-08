#!/usr/bin/env python
"""Convert COMPOD reconstruction PLY meshes to OFF, preserving n-gons.

The reconstruction PLY written by ``run_psdr_compod.py`` is a polygonal mesh:
faces are the planar boundary faces of the convex polyhedra produced by the
BSP tree, so most of them are 3..21-gons.  This converter writes them straight
into OFF (which supports arbitrary simple polygons) instead of blindly
triangulating everything.

Face handling
-------------
* simple polygon (convex or concave), no holes -> emitted as-is (n-gon kept)
* convex point set whose vertex order is scrambled -> re-ordered around the
  centroid and emitted as an n-gon
* self-intersecting / degenerate / holed polygon -> triangulated with a
  constrained Delaunay filter, with a centroid-fan fallback
* unreadable faces (bad indices, zero area) -> dropped and reported

Output layout::

    <out>/anhui/<tag>.off
    <out>/moganshan/<tag>.off
    <out>/manifest.csv

Filenames match the source OFF files 1:1, so the two trees can be loaded into
a viewer side by side for comparison.

``--watch`` keeps running while the parallel batch is still producing PLYs and
converts each newly finished model; it stops when ``progress.json`` reports
that all jobs are done (or when ``--watch-timeout`` expires).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import triangulate
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO / "data" / "oblique" / "result" / "vector_p_geometry_20261107"
DEFAULT_SRC = REPO / "data" / "oblique" / "vector_p_geometry_20261107"
DEFAULT_OUT = REPO / "data" / "oblique" / "off_recon"

AREA_EPS = 1e-12          # 2D area below this is treated as degenerate
COORD_TOL = 1e-6          # tolerance for mapping triangulation points back
MANIFEST_FIELDS = [
    "dataset", "off_file", "source_off", "src_v", "src_f",
    "out_v", "out_f", "ngon_kept", "ngon_reordered",
    "triangulated_faces", "triangles", "dropped_faces",
    "max_ngon", "ply",
]


# --------------------------------------------------------------------------- #
# PLY input
# --------------------------------------------------------------------------- #
def read_ply(path: Path):
    """Read the ASCII PLY produced by the pipeline.  Returns (V, F)."""
    raw = path.read_bytes()
    marker = raw.find(b"end_header")
    if marker < 0:
        raise ValueError("no end_header (partial write?)")
    eol = raw.find(b"\n", marker)
    eol = len(raw) if eol < 0 else eol + 1
    header = raw[:marker].decode("ascii", "replace").splitlines()

    fmt, nv, nf = "ascii 1.0", None, None
    for line in header:
        tok = line.split()
        if tok[:2] == ["format", "ascii"]:
            fmt = "ascii " + (tok[2] if len(tok) > 2 else "1.0")
        elif len(tok) >= 3 and tok[0] == "element" and tok[1] == "vertex":
            nv = int(tok[2])
        elif len(tok) >= 3 and tok[0] == "element" and tok[1] == "face":
            nf = int(tok[2])
    if not fmt.startswith("ascii"):
        raise ValueError(f"binary PLY not supported here ({fmt})")
    if nv is None or nf is None:
        raise ValueError("missing element vertex/face in header")

    lines = raw[eol:].decode("ascii", "replace").splitlines()
    if len(lines) < nv + nf:
        raise ValueError(f"truncated body: {len(lines)} lines, need {nv + nf}")

    V = np.empty((nv, 3), dtype=np.float64)
    for i in range(nv):
        tok = lines[i].split()
        if len(tok) < 3:
            raise ValueError(f"short vertex line {i}")
        V[i] = (float(tok[0]), float(tok[1]), float(tok[2]))

    F = []
    for j in range(nf):
        tok = lines[nv + j].split()
        if not tok:
            raise ValueError(f"empty face line {j}")
        k = int(tok[0])
        ids = [int(x) for x in tok[1:1 + k]]
        if len(ids) != k:
            raise ValueError(f"face {j}: expected {k} indices, got {len(ids)}")
        F.append(ids)
    return V, F


def read_off_counts(path: Path):
    try:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            fh.readline()
            tok = fh.readline().split()
            return int(tok[0]), int(tok[1])
    except Exception:
        return "", ""


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def face_basis(P):
    """Best-fit plane normal and an orthonormal 2D basis on that plane."""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    a = np.cross(n, (0.0, 0.0, 1.0))
    if np.linalg.norm(a) < 1e-8:
        a = np.cross(n, (0.0, 1.0, 0.0))
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    Q = np.column_stack(((P - c) @ a, (P - c) @ b))
    return n, Q


def newell(P):
    """Newell's normal; sign is meaningful even for slightly noisy rings."""
    q = np.roll(P, -1, axis=0)
    return np.array([
        np.sum((P[:, 1] - q[:, 1]) * (P[:, 2] + q[:, 2])),
        np.sum((P[:, 2] - q[:, 2]) * (P[:, 0] + q[:, 0])),
        np.sum((P[:, 0] - q[:, 0]) * (P[:, 1] + q[:, 1])),
    ])


def clean_ring(idx):
    """Drop consecutive duplicates and a closing duplicate."""
    out = []
    for i in idx:
        if not out or i != out[-1]:
            out.append(i)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        parts = []
        for g in geom.geoms:
            parts.extend(polygon_parts(g))
        return parts
    return []


def triangulate_face(Q, poly, ref_normal, V, idx, verts, faces, stats):
    """Triangulate a holed / self-intersecting / degenerate face.

    Uses Delaunay over the ring vertices and keeps only triangles fully inside
    the (repaired) polygon, so holes are respected.  Coordinates are mapped
    back to the original vertex indices with a KD-tree.
    """
    geoms = polygon_parts(make_valid(poly) if not poly.is_valid else poly)
    if not geoms:
        # Last resort: convex hull of the point set as a single n-gon.
        hull = MultiPoint(Q).convex_hull
        if hull.geom_type == "Polygon" and hull.area > AREA_EPS:
            ring = list(hull.exterior.coords)[:-1]
            tree = cKDTree(Q)
            dist, loc = tree.query(np.asarray(ring))
            if np.max(dist) < COORD_TOL:
                ring_idx = [idx[int(j)] for j in loc]
                faces.append(ring_idx)
                stats["triangulated_faces"] += 1
                stats["triangles"] += 1
                return True
        return False

    tree = cKDTree(Q)
    emitted = 0
    for g in geoms:
        rings = [g.exterior] + list(g.interiors)
        pts = [tuple(c) for r in rings for c in list(r.coords)[:-1]]
        if len(pts) < 3:
            continue
        padded = g.buffer(1e-9)
        for tri in triangulate(MultiPoint(pts)):
            if tri.area <= AREA_EPS or not padded.covers(tri):
                continue
            coords = np.asarray(list(tri.exterior.coords)[:-1])
            dist, loc = tree.query(coords)
            if np.max(dist) > COORD_TOL:
                continue
            tri_idx = [idx[int(j)] for j in loc]
            if len(set(tri_idx)) != 3:
                continue
            normal = np.cross(V[tri_idx[1]] - V[tri_idx[0]],
                             V[tri_idx[2]] - V[tri_idx[0]])
            if np.dot(normal, ref_normal) < 0:
                tri_idx[1], tri_idx[2] = tri_idx[2], tri_idx[1]
            faces.append(tri_idx)
            emitted += 1
    if emitted:
        stats["triangulated_faces"] += 1
        stats["triangles"] += emitted
        return True

    # Fan fallback: add a centroid vertex (simple polygons without holes only).
    g = geoms[0]
    if len(g.interiors) == 0:
        ring = list(g.exterior.coords)[:-1]
        dist, loc = tree.query(np.asarray(ring))
        if np.max(dist) < COORD_TOL and len(loc) >= 3:
            c = V[idx].mean(axis=0)
            ci = len(verts)
            verts.append(c)
            ring_idx = [idx[int(j)] for j in loc]
            for a, b in zip(ring_idx, ring_idx[1:] + ring_idx[:1]):
                normal = np.cross(V[b] - c, V[a] - c)
                tri = [ci, b, a] if np.dot(normal, ref_normal) >= 0 else [ci, a, b]
                faces.append(tri)
                emitted += 1
            stats["triangulated_faces"] += 1
            stats["triangles"] += emitted
            return True
    return False


def convert_faces(V, F):
    verts = [np.asarray(v, dtype=np.float64) for v in V]
    faces = []
    stats = Counter()

    for raw_idx in F:
        if len(raw_idx) < 3 or max(raw_idx) >= len(V) or min(raw_idx) < 0:
            stats["dropped_faces"] += 1
            continue
        idx = clean_ring(raw_idx)
        if len(idx) < 3 or len(set(idx)) != len(idx):
            stats["dropped_faces"] += 1
            continue

        P = V[idx]
        n, Q = face_basis(P)
        poly = Polygon(Q)
        ref = newell(P)
        if np.linalg.norm(ref) < 1e-12:
            ref = n

        if poly.is_valid and poly.is_simple and poly.area > AREA_EPS \
                and len(poly.interiors) == 0:
            faces.append(idx)
            stats["ngon_kept"] += 1
            stats["max_ngon"] = max(stats["max_ngon"], len(idx))
            continue

        # Scrambled but convex ring -> restore boundary order, keep the n-gon.
        order = np.argsort(np.arctan2(Q[:, 1], Q[:, 0]))
        Q2 = Q[order]
        poly2 = Polygon(Q2)
        hull = MultiPoint(Q).convex_hull
        if (poly2.is_valid and poly2.is_simple and poly2.area > AREA_EPS
                and hull.area > AREA_EPS
                and (hull.area - poly2.area) / hull.area < 1e-6):
            cand = [idx[int(k)] for k in order]
            if np.dot(newell(V[cand]), ref) < 0:
                cand.reverse()
            faces.append(cand)
            stats["ngon_reordered"] += 1
            stats["max_ngon"] = max(stats["max_ngon"], len(cand))
            continue

        if not triangulate_face(Q, poly, ref, V, idx, verts, faces, stats):
            stats["dropped_faces"] += 1

    # Compact away any unreferenced vertices (and renumber appended centroids).
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    V_out = np.asarray([verts[i] for i in used], dtype=np.float64)
    F_out = [[remap[i] for i in f] for f in faces]
    return V_out, F_out, stats


# --------------------------------------------------------------------------- #
# OFF output
# --------------------------------------------------------------------------- #
def write_off(path: Path, V, F):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write("OFF\n")
        fh.write(f"{len(V)} {len(F)} 0\n")
        for x, y, z in V:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for f in F:
            fh.write(str(len(f)) + " " + " ".join(str(i) for i in f) + "\n")
    os.replace(tmp, path)


def describe_off(off: Path, ply: Path, dataset: str, tag: str,
                 src_root: Path):
    """Build an index row for an OFF that is already up to date on disk."""
    with open(off, "r", encoding="ascii", errors="replace") as fh:
        if fh.readline().strip().upper() != "OFF":
            raise ValueError("bad OFF header")
        nv, nf = map(int, fh.readline().split()[:2])
        for _ in range(nv):
            fh.readline()
        sizes = []
        for _ in range(nf):
            tok = fh.readline().split()
            if tok:
                sizes.append(int(tok[0]))
    src = src_root / dataset / f"{tag}.off"
    src_v, src_f = read_off_counts(src)
    return {
        "dataset": dataset,
        "off_file": f"{dataset}/{tag}.off",
        "source_off": str(src).replace("\\", "/") if src.exists() else "",
        "src_v": src_v, "src_f": src_f,
        "out_v": nv, "out_f": nf,
        "ngon_kept": nf, "ngon_reordered": 0,
        "triangulated_faces": 0, "triangles": 0, "dropped_faces": 0,
        "max_ngon": max(sizes) if sizes else 0,
        "ply": str(ply).replace("\\", "/"),
    }


def convert_one(ply: Path, dataset: str, tag: str, out_root: Path,
                src_root: Path, force: bool):
    off = out_root / dataset / f"{tag}.off"
    if off.is_file() and not force and off.stat().st_mtime >= ply.stat().st_mtime:
        return None
    V, F = read_ply(ply)
    Vo, Fo, st = convert_faces(V, F)
    write_off(off, Vo, Fo)
    src = src_root / dataset / f"{tag}.off"
    src_v, src_f = read_off_counts(src)
    row = {
        "dataset": dataset,
        "off_file": f"{dataset}/{tag}.off",
        "source_off": str(src).replace("\\", "/") if src.exists() else "",
        "src_v": src_v, "src_f": src_f,
        "out_v": len(Vo), "out_f": len(Fo),
        "ngon_kept": st.get("ngon_kept", 0),
        "ngon_reordered": st.get("ngon_reordered", 0),
        "triangulated_faces": st.get("triangulated_faces", 0),
        "triangles": st.get("triangles", 0),
        "dropped_faces": st.get("dropped_faces", 0),
        "max_ngon": st.get("max_ngon", 0),
        "ply": str(ply).replace("\\", "/"),
    }
    return row


# --------------------------------------------------------------------------- #
# Batch orchestration
# --------------------------------------------------------------------------- #
def ready_tags(result_root: Path):
    s = result_root / "summary.csv"
    tags = set()
    if s.is_file():
        try:
            with open(s, "r", encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("status") in ("ok", "skipped"):
                        tags.add((r["dataset"], r["tag"]))
        except Exception:
            pass
    return tags


def batch_state(result_root: Path):
    p = result_root / "progress.json"
    if not p.is_file():
        return 0, 0
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return int(d.get("done", 0)), int(d.get("total", 0))
    except Exception:
        return 0, 0


def read_manifest(path):
    rows = {}
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[(r["dataset"], Path(r["off_file"]).stem)] = r
    return rows


def write_manifest(path, rows):
    """Atomic write.  A viewer may hold manifest.csv open on Windows, so a
    guaranteed-writable sibling (``*_live.csv``) is always kept in sync and
    returned as the authoritative index when the main file is locked."""
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS)
    w.writeheader()
    for key in sorted(rows):
        w.writerow({k: rows[key].get(k, "") for k in MANIFEST_FIELDS})
    data = buf.getvalue()

    live = path.with_name(path.stem + "_live.csv")
    try:
        with open(live, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(data)
    except OSError as exc:
        print(f"[warn] cannot write {live.name}: {exc}", flush=True)

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return path
    except OSError as exc:
        print(f"[warn] {path.name} locked ({type(exc).__name__}); "
              f"using {live.name} as the live index", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return live


def convert_pass(result_root, out_root, src_root, datasets, force, limit, quiet):
    ready = ready_tags(result_root)
    errors = []
    candidates = []
    for ds in datasets:
        for ply in sorted((result_root / ds).glob("*/*_psdr_surface.ply")):
            tag = ply.parent.name
            # Only trust a file the batch marked finished, or one that has
            # been sitting on disk long enough to rule out a partial write.
            age = time.time() - ply.stat().st_mtime
            if (ds, tag) not in ready and age < 15:
                continue
            candidates.append((ds, tag, ply))
    if limit:
        candidates = candidates[:limit]

    manifest = out_root / "manifest.csv"
    live = out_root / "manifest_live.csv"
    rows, readable = {}, False
    for candidate in (manifest, live):
        try:
            r = read_manifest(candidate)
            readable = True
            if len(r) > len(rows):        # keep the most complete index
                rows = r
        except OSError as exc:
            print(f"[warn] cannot read {candidate.name}: {exc}", flush=True)
    if not readable:
        print("[warn] no readable index; OFF conversion continues, "
              "index will be rebuilt in a later pass", flush=True)

    done = 0
    for ds, tag, ply in candidates:
        try:
            row = convert_one(ply, ds, tag, out_root, src_root, force)
        except Exception as exc:
            errors.append((ds, tag, f"{type(exc).__name__}: {exc}"))
            continue
        if row is not None:
            rows[(ds, tag)] = row
            done += 1
            if not quiet:
                print(f"[off] {ds}/{tag}: V={row['out_v']} F={row['out_f']} "
                      f"ngon={row['ngon_kept']} reordered={row['ngon_reordered']} "
                      f"tri={row['triangulated_faces']} drop={row['dropped_faces']}",
                      flush=True)
        elif (ds, tag) not in rows:
            # OFF exists but its row was lost when an earlier index write failed.
            try:
                rows[(ds, tag)] = describe_off(
                    out_root / ds / f"{tag}.off", ply, ds, tag, src_root)
            except Exception as exc:
                errors.append((ds, tag, f"index: {type(exc).__name__}: {exc}"))

    out_root.mkdir(parents=True, exist_ok=True)
    # Never let an index-write failure abort the conversion loop / watcher.
    # If neither index could be read, skip writing to avoid truncating it.
    if readable:
        try:
            write_manifest(manifest, rows)
        except Exception as exc:
            print(f"[warn] index write failed ({type(exc).__name__}: {exc}); "
                  f"OFF files are unaffected", flush=True)
    return len(candidates), done, len(rows), errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", action="append", default=None,
                    help="repeatable; default anhui + moganshan")
    ap.add_argument("--force", action="store_true", help="rewrite existing OFF")
    ap.add_argument("--limit", type=int, default=0, help="only first N (debug)")
    ap.add_argument("--watch", action="store_true",
                    help="wait for the parallel batch, converting as it goes")
    ap.add_argument("--watch-interval", type=float, default=60.0)
    ap.add_argument("--watch-timeout", type=float, default=7200.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    datasets = a.dataset or ["anhui", "moganshan"]
    if not a.result_root.is_dir():
        print(f"missing result root: {a.result_root}", file=sys.stderr)
        return 2
    for ds in datasets:
        (a.out / ds).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cand, done, total_rows, errors = convert_pass(
        a.result_root, a.out, a.src_root, datasets,
        a.force, a.limit, a.quiet)
    print(f"[pass] candidates={cand} converted={done} manifest_rows={total_rows} "
          f"errors={len(errors)}", flush=True)
    for e in errors[:10]:
        print(f"  ! {e[0]}/{e[1]}: {e[2]}", flush=True)

    if a.watch:
        last_done, last_change = -1, time.time()
        while time.time() - t0 < a.watch_timeout:
            d, tot = batch_state(a.result_root)
            if tot and d >= tot:
                print(f"[watch] batch done {d}/{tot}", flush=True)
                break
            if d != last_done:
                last_done, last_change = d, time.time()
            elif time.time() - last_change > 1800:
                print(f"[watch] no progress for 30min at {d}/{tot}, stopping",
                      file=sys.stderr, flush=True)
                break
            time.sleep(a.watch_interval)
            cand, done, total_rows, errors = convert_pass(
                a.result_root, a.out, a.src_root, datasets,
                False, 0, True)
            if done:
                print(f"[watch] +{done} converted, total {total_rows} "
                      f"(batch {d}/{tot})", flush=True)
        # final sweep, also picks up models that finished between checks
        cand, done, total_rows, errors = convert_pass(
            a.result_root, a.out, a.src_root, datasets, False, 0, a.quiet)
        d, tot = batch_state(a.result_root)
        print(f"[final] batch={d}/{tot} candidates={cand} converted={done} "
              f"manifest_rows={total_rows} elapsed={time.time()-t0:.0f}s",
              flush=True)
        for e in errors[:20]:
            print(f"  ! {e[0]}/{e[1]}: {e[2]}", flush=True)

    print(f"[out] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
