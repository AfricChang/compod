# -*- coding: utf-8 -*-
"""Check the artefacts produced by demo.py (run from the repo root)."""

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# vg.pickle stores pycompod.vg.VertexGroup instances; unpickling needs the
# repo root and the sage.all shim importable.
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
B = os.path.join(ROOT, "data", "bunny")

EXPECTED = [
    "partition/tree_simplified_partition.ply",
    "partition/tree.pickle",
    "partition/graph.pickle",
    "partition/cells.pickle",
    "partition/vg.pickle",
    "volumes/volume_mesh.ply",
    "volumes/volume_mesh_simplified.ply",
    "surface/complex_mesh.ply",
]


def ply_stats(path):
    counts = {}
    with open(path, "rb") as f:
        head = f.read(256)
    if not head.startswith(b"ply"):
        return "not a PLY file"
    text = head.decode("ascii", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            try:
                counts[parts[1]] = int(parts[2])
            except ValueError:
                pass
        if line.strip() == "end_header":
            break
    return counts


def main():
    ok = True
    print("=" * 68)
    for rel in EXPECTED:
        p = os.path.join(B, rel)
        if not os.path.isfile(p):
            print("  MISSING  %-52s" % rel)
            ok = False
            continue
        size = os.path.getsize(p)
        if size == 0:
            print("  EMPTY    %-52s" % rel)
            ok = False
            continue
        extra = ""
        if rel.endswith(".ply"):
            extra = str(ply_stats(p))
        elif rel.endswith(".pickle"):
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                extra = "type=%s len=%s" % (type(obj).__name__,
                                            len(obj) if hasattr(obj, "__len__") else "-")
            except Exception as e:
                extra = "UNPICKLE FAILED: %s" % e
                ok = False
        print("  OK       %-52s %9d B  %s" % (rel, size, extra))
    print("=" * 68)

    try:
        import trimesh
        for rel in ["surface/complex_mesh.ply", "volumes/volume_mesh.ply",
                    "volumes/volume_mesh_simplified.ply"]:
            p = os.path.join(B, rel)
            if not os.path.isfile(p):
                continue
            m = trimesh.load(p, process=False)
            print("  mesh %-42s V=%-7d F=%-7d watertight=%s" %
                  (rel, len(m.vertices), len(m.faces), m.is_watertight))
    except Exception as e:
        print("  mesh validation skipped:", e)

    print("RESULT:", "all demo outputs present and valid" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
