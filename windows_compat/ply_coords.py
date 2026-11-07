"""Coordinate-frame helpers for the oblique-building pipeline.

``off_to_pointcloud.py`` re-centres every model on its bounding-box centre
before feeding the exact cddlib kernel (large absolute coordinates would make
the Fraction numerators explode).  The reconstructed PLY therefore lives
around the origin instead of next to the source OFF.  ``run_psdr_compod.py``
translates it back with the centre recorded during sampling.  This module
provides:

  * ``off_bbox_center`` - stream an ASCII OFF and return its bbox centre
    (same definition as the centring in ``off_to_pointcloud.py``).
  * ``shift_ascii_ply`` - translate the vertices of an ASCII PLY back into
    the source coordinate frame.  Idempotent: a ``comment compod_offset``
    line in the header records that the offset has already been applied.
  * ``out_stem`` - the OFF -> result-file naming rule.
"""

from __future__ import annotations

import os
import re

_OFFSET_COMMENT = "comment compod_offset"


def out_stem(off_name: str) -> str:
    """smid_001.off -> sm1 ; smid_017__attempt_003.off -> sm17_attempt_003"""
    base = os.path.splitext(off_name)[0]
    m = re.search(r"(\d+)", base)
    stem = "sm" + str(int(m.group(1))) if m else re.sub(r"[^A-Za-z0-9_-]", "_", base)
    tail = base[m.end():] if m else ""
    tail = re.sub(r"[^A-Za-z0-9_-]", "_", tail).strip("_")
    return stem + (("_" + tail) if tail else "")


def off_bbox_center(path: str):
    """Return ((cx,cy,cz), lo, hi) of an ASCII OFF without loading the mesh."""
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        counts = None
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s.upper().startswith("OFF"):
                toks = s.split()
                # Some OFF writers put the counts on the magic line itself.
                if len(toks) >= 4:
                    try:
                        counts = [int(t) for t in toks[1:4]]
                    except ValueError:
                        pass
                break
        else:
            raise ValueError("not an OFF file: " + path)
        if counts is None:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                counts = [int(t) for t in s.split()[:3]]
                break
        if counts is None or len(counts) < 2:
            raise ValueError("missing OFF counts: " + path)
        nv = counts[0]

        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        seen = 0
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 3:
                continue
            for k in range(3):
                v = float(toks[k])
                if v < lo[k]:
                    lo[k] = v
                if v > hi[k]:
                    hi[k] = v
            seen += 1
            if seen == nv:
                break
        if seen != nv:
            raise ValueError(f"expected {nv} vertices, got {seen}: {path}")
    return [(lo[k] + hi[k]) / 2.0 for k in range(3)], lo, hi


def shift_ascii_ply(path: str, offset, decimals: int = 6) -> bool:
    """Add ``offset`` to every vertex of an ASCII PLY, in place.

    Returns True if the offset was applied, False if the file was already
    restored (header contains a ``compod_offset`` comment).
    """
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = fh.read().splitlines()
    try:
        header_end = lines.index("end_header")
    except ValueError as exc:
        raise ValueError("no end_header in " + path) from exc

    header = lines[:header_end]
    if any(ln.startswith(_OFFSET_COMMENT) for ln in header):
        return False

    nv = None
    for ln in header:
        if ln.startswith("element vertex"):
            nv = int(ln.split()[-1])
            break
    if nv is None:
        raise ValueError("no element vertex in " + path)
    if header_end + nv > len(lines):
        raise ValueError(f"truncated vertex block in {path}: "
                         f"{len(lines) - header_end - 1} < {nv}")

    for i in range(header_end + 1, header_end + 1 + nv):
        toks = lines[i].split()
        if len(toks) < 3:
            continue
        for k in range(3):
            toks[k] = f"{float(toks[k]) + offset[k]:.{decimals}f}"
        lines[i] = " ".join(toks)

    fmt = next((i for i, ln in enumerate(header) if ln.startswith("format ")), 0)
    lines.insert(fmt + 1,
                 _OFFSET_COMMENT + " " + " ".join(f"{v:.6f}" for v in offset))

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as out:
        out.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return True
