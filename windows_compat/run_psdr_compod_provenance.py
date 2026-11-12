"""独立同源导出入口；旧 GUI 入口保持不变，禁止指向正在跑批的输出文件。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "windows_compat"))

from ply_coords import shift_ascii_ply  # noqa: E402
from pycompod import PolyhedralComplex, VertexGroup  # noqa: E402
from pycompod.surface_provenance import (  # noqa: E402
    provenance_path, sha256_file, validate_coordinate_offset, write_surface_provenance,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--center-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--provenance-out", type=Path)
    parser.add_argument("--keep-classes", action="store_true")
    # 原入口的重建参数原样沿用；本次不调整拟合、图割或简化规则。
    parser.add_argument("--reg-area", type=float, default=0.8)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--save-cells", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    args.npz = args.npz.resolve(strict=True)
    args.center_json = args.center_json.resolve(strict=True)
    out = args.out.resolve()
    if out.suffix.lower() != ".ply":
        parser.error("--out must be an ASCII surface .ply")
    metadata = args.provenance_out.resolve() if args.provenance_out else provenance_path(out)
    cells = out.with_name(out.stem + "_volume_mesh.ply")
    sanitized = out.with_name(out.stem + "_compod_input.npz")
    outputs = [out, metadata] + ([cells] if args.save_cells else [])
    # 原坐标平移函数使用固定 .tmp 名称，也必须纳入冲突和覆盖检查。
    temporary = [Path(str(path) + ".tmp") for path in
                 ([out, cells] if args.save_cells else [out])]
    writes = outputs + temporary
    if len(set(writes)) != len(writes) or any(p in (args.npz, args.center_json) for p in writes):
        parser.error("Output paths must be distinct and cannot replace inputs")
    for path in writes:
        if path.exists():
            parser.error("Refusing to overwrite output: %s" % path)

    center = validate_coordinate_offset(
        json.loads(args.center_json.read_text(encoding="utf-8"))["center"])
    # 输入始终只读；需要清空 classes 时，将副本保存在新输出旁，不覆盖原 NPZ。
    t0 = time.time()
    with np.load(args.npz, allow_pickle=False) as archive:
        data = dict(archive)
    stats = {
        "points": int(data["points"].shape[0]),
        "planes": int(data["group_parameters"].shape[0]),
    }
    npz_path = args.npz
    classes = data.get("classes")
    if not args.keep_classes and classes is not None and classes.size:
        if sanitized in writes or sanitized in (args.npz, args.center_json):
            parser.error("Sanitized NPZ path conflicts with another path")
        if sanitized.exists():
            parser.error("Refusing to overwrite sanitized NPZ: %s" % sanitized)
        sanitized.parent.mkdir(parents=True, exist_ok=True)
        data["classes"] = np.zeros(0, dtype=np.int32)
        with sanitized.open("xb") as stream:
            np.savez(stream, **data)
        npz_path = sanitized

    verbosity = logging.DEBUG if args.verbose else logging.WARNING
    vg = VertexGroup(str(npz_path), verbosity=verbosity, debug_export=False)
    cc = PolyhedralComplex(vg, device="cpu", verbosity=verbosity)
    cc.construct_partition()
    cc.add_bounding_box_planes()
    cc.label_partition(mode="normals", regularization={"area": args.reg_area})
    if not args.no_simplify:
        cc.simplify_partition_tree_based()

    # 只允许新增文件；复用原来的表面导出和坐标平移，不改变面或顶点顺序。
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if args.save_cells:
        cc.save_in_cells(out_file=str(cells))
        shift_ascii_ply(str(cells), center)
    result = cc.save_surface(out_file=str(out), triangulate=False)
    if result != 0:
        raise RuntimeError("COMPOD surface export failed")
    shift_ascii_ply(str(out), center)
    provenance = write_surface_provenance(
        cc, out, metadata, coordinate_offset=center, source_npz=npz_path)
    # 保留原始输入的身份，以区分清空 classes 后实际使用的 NPZ 副本。
    print("PSDR->COMPOD PROVENANCE %s" % json.dumps({
        **stats,
        "cells": len(cc.cells),
        "seconds": round(time.time() - t0, 1),
        "output": str(out),
        "provenance": str(metadata),
        "faces": provenance["face_count"],
        "supporting_planes": len(provenance["planes"]),
        "source_npz_sha256": sha256_file(args.npz),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
