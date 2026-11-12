"""同源导出契约测试：只使用临时目录，不启动 SDK/GUI 或读取真实业务数据。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ply_coords import shift_ascii_ply


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "isolated_surface_provenance", ROOT / "pycompod" / "surface_provenance.py")
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


class FakeGraph:
    def __init__(self):
        self.edges = [(0, 1), (0, 2), (0, 3), (1, 4)]
        self.nodes = {i: {"occupancy": i == 0} for i in range(5)}
        self._adj = {0: {
            1: {"supporting_plane_id": 0},
            2: {"supporting_plane_id": 0},
            3: {"supporting_plane_id": -1},
        }, 1: {4: {"supporting_plane_id": 0}}}

    def __getitem__(self, node):
        return self._adj[node]


class FakeComplex:
    def __init__(self):
        self.graph = FakeGraph()
        self.polygons_constructed = True
        self.vg = SimpleNamespace(
            input_planes=[[0, 0, 1, -1], [0, -1, 0, 0]],
            plane_order=[4, 1, 3, 0, 2],
            merged_plane_from_input_planes=[[1, 4]],
            epsilon=None,
        )
        self.intersections = {
            (0, 1): [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            (0, 2): [[1, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1]],
            (0, 3): [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        }

    def _get_intersection(self, a, b):
        return self.intersections[(a, b)]


def write_fake_surface(path, complex_, face_order=None):
    faces = list(complex_.intersections.values())
    if face_order is not None:
        faces = [faces[i] for i in face_order]
    vertices = sorted({tuple(point) for face in faces for point in face})
    indices = {point: i for i, point in enumerate(vertices)}
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(vertices))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("element face %d\nproperty list uchar int vertex_indices\nend_header\n" % len(faces))
        for point in vertices:
            stream.write(" ".join("%.6f" % value for value in point) + "\n")
        for face in faces:
            ring = [indices[tuple(point)] for point in face]
            stream.write("%d %s\n" % (len(ring), " ".join(str(i) for i in ring)))


class SurfaceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compod_provenance_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = self.root / "surface.ply"
        self.cc = FakeComplex()
        write_fake_surface(self.mesh, self.cc)

    def export(self, **kwargs):
        return provenance.write_surface_provenance(self.cc, self.mesh, **kwargs)

    def metadata(self):
        return provenance.provenance_path(self.mesh)

    def change_metadata(self, change):
        data = json.loads(self.metadata().read_text(encoding="utf-8"))
        change(data)
        self.metadata().write_text(json.dumps(data), encoding="utf-8")

    def test_bsp_fragments_keep_same_parent_and_skip_internal_interface(self):
        data = self.export()
        self.assertEqual(data["face_plane_ids"], [0, 0, -1])
        self.assertEqual(data["face_graph_edges"], [[0, 1], [0, 2], [0, 3]])
        self.assertEqual(data["face_count"], 3)
        self.assertEqual(provenance.load_surface_provenance(self.mesh), data)

    def test_sorted_merged_group_ids_are_not_internal_plane_ids(self):
        data = self.export()
        self.assertEqual(data["planes"]["0"]["source_group_ids"], [1, 2])
        self.assertEqual(data["planes"]["0"]["source_group_status"], "exact")

    def test_bounding_plane_keeps_signed_id_and_has_no_source_groups(self):
        plane = self.export()["planes"]["-1"]
        self.assertEqual(plane["kind"], "bounding_box")
        self.assertEqual(plane["source_group_ids"], [])

    def test_plane_clustering_does_not_claim_exact_npz_row_ids(self):
        self.cc.vg.epsilon = 0.1
        plane = self.export()["planes"]["0"]
        self.assertIsNone(plane["source_group_ids"])
        self.assertEqual(plane["source_group_status"], "unavailable_after_plane_clustering")

    def test_missing_optional_group_mapping_still_preserves_internal_plane(self):
        del self.cc.vg.plane_order
        data = self.export()
        self.assertEqual(data["face_plane_ids"], [0, 0, -1])
        self.assertIsNone(data["planes"]["0"]["source_group_ids"])

    def test_disabled_duplicate_merge_traces_only_the_actual_original_group(self):
        self.cc.vg.merge_duplicate_planes = False
        plane = self.export()["planes"]["0"]
        self.assertEqual(plane["source_group_ids"], [4])
        self.assertEqual(plane["source_group_status"], "exact")

    def test_mesh_and_complex_remain_unchanged(self):
        original = self.mesh.read_bytes()
        original_edges = list(self.cc.graph.edges)
        self.export()
        self.assertEqual(self.mesh.read_bytes(), original)
        self.assertEqual(self.cc.graph.edges, original_edges)

    def test_translation_updates_plane_equations_and_preserves_lineage(self):
        center = [1000.25, -300.5, 4.75]
        shift_ascii_ply(str(self.mesh), center)
        data = self.export(coordinate_offset=center)
        self.assertEqual(data["planes"]["0"]["coefficients"], [0.0, 0.0, 1.0, -5.75])
        self.assertEqual(data["planes"]["-1"]["coefficients"], [0.0, -1.0, 0.0, -300.5])
        self.assertEqual(data["coordinate_offset_from_compod"], center)
        self.assertEqual(provenance.load_surface_provenance(self.mesh), data)

    def test_translation_reproduces_both_legacy_rounding_steps(self):
        for face in self.cc.intersections.values():
            for point in face:
                point[0] += 0.12345649
        write_fake_surface(self.mesh, self.cc)
        center = [0.00000049, 0.0, 0.0]
        shift_ascii_ply(str(self.mesh), center)
        self.export(coordinate_offset=center)
        self.assertEqual(provenance.load_surface_provenance(self.mesh)["face_count"], 3)

    def test_wrong_offset_is_rejected_before_creating_sidecar(self):
        shift_ascii_ply(str(self.mesh), [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.export()
        self.assertFalse(self.metadata().exists())

    def test_reordered_faces_are_not_silently_zipped_to_old_ids(self):
        write_fake_surface(self.mesh, self.cc, [2, 1, 0])
        with self.assertRaisesRegex(ValueError, "face 0 does not match"):
            self.export()
        self.assertFalse(self.metadata().exists())

    def test_changed_face_count_is_rejected(self):
        self.cc.graph.edges = self.cc.graph.edges[1:]
        with self.assertRaisesRegex(ValueError, "face count"):
            self.export()

    def test_unconstructed_complex_is_rejected(self):
        self.cc.polygons_constructed = False
        with self.assertRaisesRegex(ValueError, "save_surface"):
            self.export()

    def test_existing_metadata_is_not_overwritten(self):
        self.export()
        original = self.metadata().read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.metadata().read_bytes(), original)

    def test_mesh_cannot_be_used_as_metadata_output(self):
        original = self.mesh.read_bytes()
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.export(output_path=self.mesh)
        self.assertEqual(self.mesh.read_bytes(), original)

    def test_input_npz_identity_is_hashed_without_modifying_input(self):
        source = self.root / "input.npz"
        source.write_bytes(b"synthetic identity fixture")
        data = self.export(source_npz=source)
        self.assertEqual(data["input_npz"]["sha256"], provenance.sha256_file(source))
        self.assertEqual(source.read_bytes(), b"synthetic identity fixture")

    def test_mesh_change_during_export_is_rejected(self):
        with patch.object(provenance, "sha256_file", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "changed while"):
                self.export()
        self.assertFalse(self.metadata().exists())

    def test_stale_sidecar_rejected_even_if_only_mesh_comment_changed(self):
        self.export()
        self.mesh.write_text(self.mesh.read_text().replace(
            "format ascii 1.0", "format ascii 1.0\ncomment changed"), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            provenance.load_surface_provenance(self.mesh)

    def test_mesh_change_during_loading_is_rejected(self):
        data = self.export()
        with patch.object(provenance, "sha256_file",
                          side_effect=[data["mesh_sha256"], "changed"]):
            with self.assertRaisesRegex(ValueError, "changed while loading"):
                provenance.load_surface_provenance(self.mesh)

    def test_schema_mismatch_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data.update(schema="unknown"))
        with self.assertRaisesRegex(ValueError, "schema"):
            provenance.load_surface_provenance(self.mesh)

    def test_cross_reconstruction_id_scope_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data.update(id_scope="global"))
        with self.assertRaisesRegex(ValueError, "ID scope"):
            provenance.load_surface_provenance(self.mesh)

    def test_missing_face_parent_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data["face_plane_ids"].pop())
        with self.assertRaisesRegex(ValueError, "ID count"):
            provenance.load_surface_provenance(self.mesh)

    def test_missing_plane_table_entry_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data["planes"].pop("0"))
        with self.assertRaisesRegex(ValueError, "supporting plane ID"):
            provenance.load_surface_provenance(self.mesh)

    def test_boolean_plane_id_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data["face_plane_ids"].__setitem__(0, True))
        with self.assertRaisesRegex(ValueError, "supporting plane ID"):
            provenance.load_surface_provenance(self.mesh)

    def test_nonfinite_plane_and_offset_are_rejected(self):
        self.cc.vg.input_planes[0][3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.export()
        with self.assertRaisesRegex(ValueError, "finite"):
            provenance.validate_coordinate_offset([0.0, float("inf"), 0.0])

    def test_zero_normal_is_rejected(self):
        self.cc.vg.input_planes[0] = [0, 0, 0, 1]
        with self.assertRaisesRegex(ValueError, "zero plane normal"):
            self.export()

    def test_missing_graph_edge_mapping_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data["face_graph_edges"].pop())
        with self.assertRaisesRegex(ValueError, "graph edge count"):
            provenance.load_surface_provenance(self.mesh)

    def test_invalid_plane_record_is_rejected(self):
        self.export()
        self.change_metadata(lambda data: data["planes"].update({"0": None}))
        with self.assertRaisesRegex(ValueError, "plane record"):
            provenance.load_surface_provenance(self.mesh)

    def test_unavailable_raw_group_mapping_cannot_claim_exact_lineage(self):
        self.export()
        self.change_metadata(lambda data: data["planes"]["0"].update(source_group_ids=None))
        with self.assertRaisesRegex(ValueError, "source group status"):
            provenance.load_surface_provenance(self.mesh)

    def test_bounding_box_cannot_claim_input_groups(self):
        self.export()
        self.change_metadata(lambda data: data["planes"]["-1"].update(source_group_ids=[0]))
        with self.assertRaisesRegex(ValueError, "source group status"):
            provenance.load_surface_provenance(self.mesh)


if __name__ == "__main__":
    unittest.main()
