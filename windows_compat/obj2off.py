"""Batch-convert every OBJ under any `margin_4m` folder of the multiview
collection into ASCII OFF, grouped by dataset (anhui / moganshan).

Usage:
    python windows_compat/obj2off.py [--src SRC] [--dst DST] [--force]
    python windows_compat/obj2off.py --src <folder-of-dataset-dirs> \
        --dst data/oblique/vector_p_geometry_20261107 --layout flat --jobs 8

The converter is deliberately dependency-free (standard library only):
  * keeps the original vertex coordinate strings, so no precision is lost;
  * supports OBJ indices of the form  v, v/vt, v//vn, v/vt/vn and negatives;
  * preserves n-gons (OFF allows arbitrary polygon size), no triangulation;
  * drops materials / normals / texture coordinates (OFF has no such fields);
  * writes a manifest CSV mapping every OFF back to its source OBJ.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

DEFAULT_SRC = (
    r"E:\github\ObliqueBuilding_Fine_tuning\Data\multiview_batches"
    r"\full_execution_20261106_112608\collection\attempts"
)
DEFAULT_DST = r"E:\github\compod\data\oblique"


def _dataset_of(name: str) -> str:
    low = name.lower()
    if low.startswith("anhui"):
        return "anhui"
    if low.startswith("moganshan"):
        return "moganshan"
    return name


def find_objs(src: str, layout: str = "margin"):
    """Yield (dataset, source_obj_path) for every OBJ found under ``src``.

    layout="margin" (legacy multiview collection): only OBJ files inside a
      ``margin_4m`` folder; dataset comes from the first path component.
    layout="flat" (vector-clipped export): every direct child directory of
      ``src`` is a dataset holding the OBJ files.  If ``src`` itself holds
      OBJ files, its own name is used as the dataset.
    """
    if layout == "flat":
        subs = [d for d in sorted(os.listdir(src))
                if os.path.isdir(os.path.join(src, d))]
        if not subs:
            subs = ["."]
        for name in subs:
            sub = os.path.join(src, name)
            dataset = (_dataset_of(name) if name != "."
                       else _dataset_of(os.path.basename(os.path.normpath(src))))
            for dirpath, _dirs, filenames in os.walk(sub):
                for fn in filenames:
                    if fn.lower().endswith(".obj"):
                        yield dataset, os.path.join(dirpath, fn)
        return

    for dirpath, dirnames, filenames in os.walk(src):
        parts = dirpath.replace("/", os.sep).split(os.sep)
        if "margin_4m" not in parts:
            continue
        rel_first = os.path.relpath(dirpath, src).replace("/", os.sep).split(os.sep)[0]
        dataset = _dataset_of(rel_first)
        if dataset not in ("anhui", "moganshan"):
            continue
        for fn in filenames:
            if fn.lower().endswith(".obj"):
                yield dataset, os.path.join(dirpath, fn)


def parse_obj(path: str):
    """Return (vertex_lines, face_index_lists, stats)."""
    verts: list[str] = []
    faces: list[list[int]] = []
    skipped_faces = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v ") or line.startswith("v\t"):
                toks = line.split()
                if len(toks) < 4:
                    continue
                # keep the exact original numeric strings (first 3 coordinates)
                verts.append(" ".join(toks[1:4]))
            elif line.startswith("f ") or line.startswith("f\t"):
                toks = line.split()[1:]
                idx = []
                for tok in toks:
                    vtok = tok.split("/")[0]
                    if not vtok:
                        continue
                    try:
                        idx.append(int(vtok))
                    except ValueError:
                        idx = []
                        break
                if len(idx) >= 3:
                    faces.append(idx)
                else:
                    skipped_faces += 1
    return verts, faces, skipped_faces


def _face_indices(line: str):
    """Parse one OBJ face line -> list of 1-based (possibly negative) indices."""
    idx = []
    for tok in line.split()[1:]:
        vtok = tok.split("/")[0]
        if not vtok:
            return []
        try:
            idx.append(int(vtok))
        except ValueError:
            return []
    return idx if len(idx) >= 3 else []


def _scan(path: str):
    """First pass: count vertices and valid faces (constant memory)."""
    nv = nf = skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("v ") or line.startswith("v\t"):
                if len(line.split()) >= 4:
                    nv += 1
            elif line.startswith("f ") or line.startswith("f\t"):
                if _face_indices(line):
                    nf += 1
                else:
                    skipped += 1
    return nv, nf, skipped


def convert_obj_to_off(src: str, dst_path: str) -> tuple[int, int, int]:
    """Two-pass streaming OBJ -> OFF converter, O(1) memory.

    Pass 1 counts vertices/faces so the OFF header and negative (relative)
    indices can be resolved; pass 2 writes header + all vertex lines; pass 3
    appends the face block. OFF only requires the first nv lines after the
    header to be vertices and the next nf to be faces, so interleaved OBJ
    input is fine. No temporary files are created or deleted.
    """
    nv, nf, skipped = _scan(src)
    if nv == 0 or nf == 0:
        raise ValueError("no vertices or no faces")

    # Pass 2: header + all vertex lines. OFF requires the first nv lines after
    # the header to be vertices, so we deliberately separate the two blocks.
    with open(src, "r", encoding="utf-8", errors="replace") as fh, \
         open(dst_path, "w", encoding="ascii", newline="\n") as out:
        out.write(f"OFF\n{nv} {nf} 0\n")
        for raw in fh:
            line = raw.strip()
            if line.startswith("v ") or line.startswith("v\t"):
                toks = line.split()
                if len(toks) >= 4:
                    out.write(" ".join(toks[1:4]) + "\n")

    # Pass 3: append the face block directly, no sidecar / no deletion.
    with open(src, "r", encoding="utf-8", errors="replace") as fh, \
         open(dst_path, "a", encoding="ascii", newline="\n") as out:
        for raw in fh:
            line = raw.strip()
            if not (line.startswith("f ") or line.startswith("f\t")):
                continue
            idx = _face_indices(line)
            if not idx:
                continue
            resolved = [(i - 1) if i > 0 else (nv + i) for i in idx]
            if any(i < 0 or i >= nv for i in resolved):
                raise ValueError(f"face index out of range: {line[:80]}")
            out.write(str(len(resolved)) + " " + " ".join(map(str, resolved)) + "\n")
    return nv, nf, skipped


def write_off(dst_path: str, verts, faces) -> tuple[int, int]:
    """Small in-memory helper (used by the smoke test)."""
    nv = len(verts)
    nf = len(faces)
    with open(dst_path, "w", encoding="ascii", newline="\n") as out:
        out.write("OFF\n")
        out.write(f"{nv} {nf} 0\n")
        out.write("\n".join(verts))
        out.write("\n")
        for face in faces:
            resolved = [(i - 1) if i > 0 else (nv + i) for i in face]
            if any(i < 0 or i >= nv for i in resolved):
                raise ValueError(f"face index out of range in {dst_path}: {face}")
            out.write(str(len(resolved)) + " " + " ".join(map(str, resolved)) + "\n")
    return nv, nf


def unique_name(dataset: str, obj_path: str, used: set[str]) -> str:
    """smid_017.obj -> smid_017.off; on collision append attempt tag."""
    base = os.path.splitext(os.path.basename(obj_path))[0]
    name = base + ".off"
    if name not in used:
        used.add(name)
        return name
    # encode the attempt folder so the duplicate stays traceable
    parts = obj_path.replace("/", os.sep).split(os.sep)
    attempt = next((p for p in parts if p.startswith("attempt_")), "dup")
    name = f"{base}__{attempt}.off"
    n = 2
    while name in used:
        name = f"{base}__{attempt}_{n}.off"
        n += 1
    used.add(name)
    return name


def convert_one(task):
    """Worker for --jobs > 1: convert one OBJ and return (manifest_row, error)."""
    dataset, name, obj_path, out_path, force = task
    if os.path.exists(out_path) and not force:
        return (dataset, name, obj_path, "", "", "skipped"), None
    try:
        nv, nf, skipped_faces = convert_obj_to_off(obj_path, out_path)
        return (dataset, name, obj_path, nv, nf,
                f"skipped_faces={skipped_faces}"), None
    except Exception as exc:
        return (dataset, name, obj_path, "", "", f"ERROR: {exc}"), str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--layout", choices=("margin", "flat"), default="margin",
                    help="margin = legacy margin_4m tree; flat = one folder per dataset")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel converter processes (1 = serial, legacy behaviour)")
    ap.add_argument("--force", action="store_true", help="overwrite existing OFF files")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        print(f"source not found: {src}", file=sys.stderr)
        return 2
    os.makedirs(dst, exist_ok=True)

    jobs = sorted(find_objs(src, args.layout), key=lambda t: (t[0], t[1].lower()))
    by_ds: dict[str, list[str]] = defaultdict(list)
    for ds, p in jobs:
        by_ds[ds].append(p)
    print(f"source : {src}")
    print(f"target : {dst}")
    print(f"layout : {args.layout}   jobs: {args.jobs}")
    print(f"found  : {len(jobs)} OBJ  ({', '.join(f'{k}={len(v)}' for k, v in sorted(by_ds.items()))})")

    used: dict[str, set[str]] = defaultdict(set)
    tasks = []
    for dataset, obj_path in jobs:
        out_dir = os.path.join(dst, dataset)
        os.makedirs(out_dir, exist_ok=True)
        name = unique_name(dataset, obj_path, used[dataset])
        tasks.append((dataset, name, obj_path,
                      os.path.join(out_dir, name), args.force))

    rows = []
    stats = Counter()
    t0 = time.time()

    def record(result, i):
        row, err = result
        rows.append(row)
        note = row[-1]
        if note == "skipped":
            stats["skipped"] += 1
        elif note.startswith("ERROR"):
            stats["failed"] += 1
            print(f"[{i}/{len(tasks)}] FAIL {os.path.basename(row[2])}: {err}",
                  flush=True)
        else:
            stats["converted"] += 1
            stats["vertices"] += int(row[3])
            stats["faces"] += int(row[4])
        if i % 20 == 0 or i == len(tasks):
            print(f"[{i}/{len(tasks)}] {time.time()-t0:.1f}s  "
                  f"verts={stats['vertices']:,} faces={stats['faces']:,}", flush=True)

    if args.jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(convert_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                record(fut.result(), i)
    else:
        for i, t in enumerate(tasks, 1):
            record(convert_one(t), i)

    rows.sort(key=lambda r: (r[0], r[1]))
    manifest = os.path.join(dst, "manifest.csv")
    with open(manifest, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "off_file", "source_obj", "n_vertices", "n_faces", "note"])
        w.writerows(rows)

    print("-" * 60)
    print(f"converted={stats['converted']}  skipped={stats['skipped']}  failed={stats['failed']}")
    print(f"total vertices={stats['vertices']:,}  faces={stats['faces']:,}  {time.time()-t0:.1f}s")
    for ds in sorted(by_ds):
        print(f"  {dst}\\{ds}  ->  {len(used[ds])} OFF files")
    print(f"manifest: {manifest}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
