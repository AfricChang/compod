"""导出表面面片到 COMPOD 支撑平面的对应关系，不改动已有表面文件。

此模块不由 pycompod.__init__ 自动导入；正在运行的旧入口不会启用它。
只接受当前 python/python_exact 后端导出的、未经拓扑变换的 ASCII PLY。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "compod-surface-provenance-v1"
# 沿用 export_complex.write_surface_to_ply 和 ply_coords.shift_ascii_ply 的输出精度。
# 这是文件序列化契约，不是平面拟合或共面归并容差。
LEGACY_PLY_DECIMAL_DIGITS = 6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(mesh_path):
    return Path(mesh_path).with_suffix(".provenance.json")


def _finite_vector(values, size, name):
    values = [float(x) for x in values]
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError("%s must contain %d finite values" % (name, size))
    return values


def validate_coordinate_offset(values):
    return _finite_vector(values, 3, "offset")


def _read_surface_ply(path):
    """只读取表面几何；不猜测二进制 PLY 或转换后的 OFF 的面序。"""
    with Path(path).open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Expected PLY surface")
        nv, nf, fmt, element = None, None, None, None
        vertex_properties, face_properties = [], []
        for line in stream:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "end_header":
                break
            if tokens[0] == "format":
                fmt = tokens[1:]
            elif tokens[0] == "element":
                element = tokens[1]
                if element == "vertex":
                    nv = int(tokens[2])
                elif element == "face":
                    nf = int(tokens[2])
                else:
                    raise ValueError("Unsupported PLY element: %s" % element)
            elif tokens[0] == "property":
                if element == "vertex":
                    vertex_properties.append(tokens[1:])
                elif element == "face":
                    face_properties.append(tokens[1:])
        else:
            raise ValueError("Missing PLY end_header")
        if fmt != ["ascii", "1.0"]:
            raise ValueError("Only ASCII PLY 1.0 is supported")
        if nv is None or nf is None or nv < 0 or nf <= 0:
            raise ValueError("Invalid surface vertex/face counts")
        if [p[-1] for p in vertex_properties[:3]] != ["x", "y", "z"]:
            raise ValueError("Expected x/y/z as the first vertex properties")
        if not face_properties or face_properties[0] != [
                "list", "uchar", "int", "vertex_indices"]:
            raise ValueError("Unsupported PLY face index property")
        vertices = []
        for _ in range(nv):
            tokens = stream.readline().split()
            if len(tokens) < 3:
                raise ValueError("Truncated PLY vertex block")
            vertices.append(_finite_vector(tokens[:3], 3, "vertex"))
        faces = []
        for _ in range(nf):
            tokens = stream.readline().split()
            if not tokens:
                raise ValueError("Truncated PLY face block")
            size = int(tokens[0])
            if size < 3 or len(tokens) < size + 1:
                raise ValueError("Invalid PLY face ring")
            face = [int(x) for x in tokens[1:size + 1]]
            if any(i < 0 or i >= nv for i in face):
                raise ValueError("PLY vertex index out of range")
            faces.append(face)
    return vertices, faces


def _serialized_point(point, offset):
    # 必须先复现中心化 PLY 的舍入，再复现平移后的舍入，不能直接对原点相加。
    result = []
    for value, delta in zip(point, offset):
        first = float(format(float(value), ".%df" % LEGACY_PLY_DECIMAL_DIGITS))
        result.append(float(format(first + delta, ".%df" % LEGACY_PLY_DECIMAL_DIGITS)))
    return tuple(result)


def _source_groups(vg, plane_id):
    if plane_id < 0:
        return [], "bounding_box"
    if getattr(vg, "epsilon", None) is not None:
        # 平面聚类不保留完整原始组谱系，不能把聚类后的编号冒充 NPZ 原始行号。
        return None, "unavailable_after_plane_clustering"
    order = getattr(vg, "plane_order", None)
    if getattr(vg, "merge_duplicate_planes", True) is False:
        # 不启用合并时，原库的同名映射只是等参数平面的评估表，不代表真实合并。
        if order is None:
            return None, "unavailable"
        return [int(order[plane_id])], "exact"
    merged = getattr(vg, "merged_plane_from_input_planes", None)
    if merged is None or order is None:
        return None, "unavailable"
    groups = sorted({int(order[int(i)]) for i in merged[plane_id]})
    return groups, "exact"


def build_surface_provenance(complex_, mesh_path, coordinate_offset=None, source_npz=None):
    """针对同一存活 cc 对象刚导出的 PLY 生成元数据，并逐面验证面序。

    coordinate_offset 必须是实际传给 shift_ascii_ply 的原始中心值；不提供表示
    PLY 未平移。不能从旧 PLY 重新拟合平面来伪造同源关系。
    """
    mesh_path = Path(mesh_path)
    offset = _finite_vector(
        [0.0, 0.0, 0.0] if coordinate_offset is None else coordinate_offset, 3, "offset")
    if not getattr(complex_, "polygons_constructed", False):
        raise ValueError("Call save_surface before exporting provenance")
    mesh_hash = sha256_file(mesh_path)
    vertices, faces = _read_surface_ply(mesh_path)
    interfaces = [
        (a, b) for a, b in complex_.graph.edges
        if complex_.graph.nodes[a]["occupancy"] != complex_.graph.nodes[b]["occupancy"]
    ]
    if len(interfaces) != len(faces):
        raise ValueError("Surface face count does not match COMPOD interfaces")
    plane_ids, graph_edges = [], []
    for face_id, ((a, b), face) in enumerate(zip(interfaces, faces)):
        edge = complex_.graph[a][b]
        plane_id = int(edge["supporting_plane_id"])
        intersection = complex_._get_intersection(a, b)
        expected = sorted(_serialized_point(point, offset) for point in intersection)
        actual = sorted(tuple(vertices[i]) for i in face)
        if actual != expected:
            raise ValueError("Surface face %d does not match its COMPOD interface" % face_id)
        plane_ids.append(plane_id)
        graph_edges.append([int(a), int(b)])

    planes = {}
    for plane_id in sorted(set(plane_ids)):
        coefficients = _finite_vector(complex_.vg.input_planes[plane_id], 4, "plane")
        if not any(coefficients[:3]):
            raise ValueError("Invalid zero plane normal")
        # x_mesh = x_compod + offset，因此 d_mesh = d_compod - n dot offset。
        coefficients[3] -= sum(n * x for n, x in zip(coefficients[:3], offset))
        if not all(math.isfinite(x) for x in coefficients):
            raise ValueError("Non-finite translated plane")
        groups, status = _source_groups(complex_.vg, plane_id)
        planes[str(plane_id)] = {
            "kind": "bounding_box" if plane_id < 0 else "input",
            "coefficients": coefficients,
            "source_group_ids": groups,
            "source_group_status": status,
        }

    if sha256_file(mesh_path) != mesh_hash:
        raise ValueError("Surface changed while building provenance")
    result = {
        "schema": SCHEMA,
        "mesh_file": mesh_path.name,
        "mesh_sha256": mesh_hash,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "id_scope": "single_compod_reconstruction",
        "plane_equation": "a*x+b*y+c*z+d=0",
        "coordinate_frame": "mesh",
        "coordinate_offset_from_compod": offset,
        "face_plane_ids": plane_ids,
        "face_graph_edges": graph_edges,
        "planes": planes,
    }
    if source_npz is not None:
        result["input_npz"] = {
            "file": Path(source_npz).name, "sha256": sha256_file(source_npz)}
    return result


def write_surface_provenance(complex_, mesh_path, output_path=None,
                             coordinate_offset=None, source_npz=None):
    """只新增 sidecar；拒绝覆盖已有文件，不更改 mesh 或 COMPOD 对象。"""
    mesh_path = Path(mesh_path)
    output_path = provenance_path(mesh_path) if output_path is None else Path(output_path)
    if output_path.resolve() == mesh_path.resolve():
        raise ValueError("Provenance path must differ from mesh path")
    if output_path.exists():
        raise FileExistsError("Refusing to overwrite provenance: %s" % output_path)
    result = build_surface_provenance(complex_, mesh_path, coordinate_offset, source_npz)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return result


def load_surface_provenance(mesh_path, metadata_path=None):
    """检查网格配对和 ID 完整性，拒绝将错位或陈旧的 sidecar 用于归并。"""
    mesh_path = Path(mesh_path)
    metadata_path = provenance_path(mesh_path) if metadata_path is None else Path(metadata_path)
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unsupported surface provenance schema")
    if data.get("id_scope") != "single_compod_reconstruction":
        raise ValueError("Unsupported provenance ID scope")
    if data.get("mesh_sha256") != sha256_file(mesh_path):
        raise ValueError("Surface provenance mesh hash mismatch")
    vertices, faces = _read_surface_ply(mesh_path)
    if data.get("vertex_count") != len(vertices) or data.get("face_count") != len(faces):
        raise ValueError("Surface provenance mesh counts mismatch")
    ids = data.get("face_plane_ids")
    if not isinstance(ids, list) or len(ids) != len(faces):
        raise ValueError("Surface provenance plane ID count mismatch")
    planes = data.get("planes")
    if not isinstance(planes, dict):
        raise ValueError("Missing provenance plane table")
    for plane_id in ids:
        if type(plane_id) is not int or str(plane_id) not in planes:
            raise ValueError("Invalid or missing supporting plane ID")
    graph_edges = data.get("face_graph_edges")
    if not isinstance(graph_edges, list) or len(graph_edges) != len(faces):
        raise ValueError("Surface provenance graph edge count mismatch")
    if any(not isinstance(edge, list) or len(edge) != 2
           or any(type(node) is not int for node in edge) for edge in graph_edges):
        raise ValueError("Invalid surface provenance graph edge")
    if data.get("coordinate_frame") != "mesh" or data.get("plane_equation") != "a*x+b*y+c*z+d=0":
        raise ValueError("Unsupported provenance coordinate/plane convention")
    _finite_vector(data.get("coordinate_offset_from_compod", []), 3, "offset")
    for plane_id in set(ids):
        plane = planes[str(plane_id)]
        if not isinstance(plane, dict):
            raise ValueError("Invalid provenance plane record")
        coefficients = _finite_vector(plane.get("coefficients", []), 4, "plane")
        if not any(coefficients[:3]):
            raise ValueError("Invalid zero plane normal")
        kind = "bounding_box" if plane_id < 0 else "input"
        if plane.get("kind") != kind:
            raise ValueError("Invalid provenance plane kind")
        groups = plane.get("source_group_ids")
        if groups is not None and (not isinstance(groups, list) or any(
                type(group) is not int or group < 0 for group in groups)):
            raise ValueError("Invalid source group IDs")
        status = plane.get("source_group_status")
        if plane_id < 0:
            valid_source = status == "bounding_box" and groups == []
        elif status == "exact":
            valid_source = bool(groups) and groups == sorted(set(groups))
        else:
            valid_source = (status in ("unavailable", "unavailable_after_plane_clustering")
                            and groups is None)
        if not valid_source:
            raise ValueError("Inconsistent provenance source group status")
    if data["mesh_sha256"] != sha256_file(mesh_path):
        raise ValueError("Surface changed while loading provenance")
    return data
