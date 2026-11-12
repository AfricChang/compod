"""小型真实 BSP/入口对照测试；只生成临时立方体，不读取 GUI 或业务数据。"""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "windows_compat"))

from ply_coords import shift_ascii_ply  # noqa: E402
from pycompod import PolyhedralComplex, VertexGroup  # noqa: E402
from pycompod.surface_provenance import (  # noqa: E402
    load_surface_provenance, sha256_file, write_surface_provenance,
)
import run_psdr_compod_provenance as provenance_entry  # noqa: E402


def make_fixture(root):
    # 六个外表面，加一个穿过内部的切平面，再加一个重复屋面检测组。
    planes = np.array([
        [-1, 0, 0, -0.5], [1, 0, 0, -0.5],
        [0, -1, 0, -0.5], [0, 1, 0, -0.5],
        [0, 0, -1, 0], [0, 0, 1, -1],
        [1, 0, 0, 0], [0, 0, 1, -1],
    ], dtype=np.float64)
    squares = [
        [[-0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1], [-0.5, -0.5, 1]],
        [[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]],
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]],
        [[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]],
        [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
        [[0, -1, -0.5], [0, 1, -0.5], [0, 1, 1.5], [0, -1, 1.5]],
        [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
    ]
    # 法向投票使用严格面内点，只有四角的旧夹具会得到零票。
    # 保留边界点定义包围范围，同时补充规则面内点，让切平面也有立方体内部支撑。
    parameters = np.linspace(0.0, 1.0, 5)
    groups = []
    for square in squares:
        corner = np.asarray(square, dtype=np.float64)
        groups.append(np.array([
            corner[0] + u * (corner[1] - corner[0]) + v * (corner[3] - corner[0])
            for u in parameters for v in parameters
        ]))
    counts = np.array([len(group) for group in groups], dtype=np.int32)
    path = root / "cube.npz"
    np.savez(
        path,
        points=np.concatenate(groups),
        normals=np.repeat(planes[:, :3], counts, axis=0),
        group_parameters=planes,
        group_colors=np.full((len(planes), 3), 150, dtype=np.uint8),
        group_num_points=counts,
        group_points=np.arange(sum(counts), dtype=np.int32),
        classes=np.ones(sum(counts), dtype=np.int32),
    )
    center = [1280.1234567, -7500.1234567, 15.2500004]
    center_path = root / "center.json"
    center_path.write_text(json.dumps({"center": center}), encoding="utf-8")
    return path, center_path, center


class RealSurfaceProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="compod_provenance_bsp_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.npz, cls.center_path, cls.center = make_fixture(cls.root)
        vg = VertexGroup(str(cls.npz), verbosity=logging.ERROR)
        cls.cc = PolyhedralComplex(vg, device="cpu", verbosity=logging.ERROR, n_threads=1)
        cls.cc.construct_partition()
        cls.cc.add_bounding_box_planes()
        # 单独验证导出谱系时，用解析立方体内部判定设置 occupancy，不依赖图割结果。
        for node, attributes in cls.cc.graph.nodes(data=True):
            center = [float(x) for x in cls.cc.cells[node].center()]
            inside = (-0.5 < center[0] < 0.5 and -0.5 < center[1] < 0.5
                      and 0 < center[2] < 1)
            attributes["occupancy"] = int(inside and "bounding_box" not in attributes)
        cls.cc.partition_labelled = True
        cls.mesh = cls.root / "surface.ply"
        result = cls.cc.save_surface(str(cls.mesh), triangulate=False)
        if result != 0:
            raise AssertionError("Synthetic cube surface export failed")
        shift_ascii_ply(str(cls.mesh), cls.center)
        cls.mesh_bytes = cls.mesh.read_bytes()
        cls.data = write_surface_provenance(
            cls.cc, cls.mesh, coordinate_offset=cls.center, source_npz=cls.npz)
        print("SYNTHETIC_PROVENANCE " + json.dumps({
            "face_count": cls.data["face_count"],
            "faces_per_plane": dict(Counter(cls.data["face_plane_ids"])),
            "source_groups_per_plane": {
                pid: plane["source_group_ids"] for pid, plane in cls.data["planes"].items()
            },
        }, sort_keys=True))

    def test_real_bsp_fragments_and_duplicate_input_groups_survive(self):
        counts = Counter(self.data["face_plane_ids"])
        self.assertTrue(any(pid >= 0 and count > 1 for pid, count in counts.items()))
        self.assertTrue(any(
            plane["source_group_ids"] == [5, 7]
            for plane in self.data["planes"].values()))
        self.assertEqual(load_surface_provenance(self.mesh), self.data)

    def test_metadata_export_does_not_change_legacy_mesh_bytes(self):
        self.assertEqual(self.mesh.read_bytes(), self.mesh_bytes)
        repeat = self.root / "repeat.ply"
        self.assertEqual(self.cc.save_surface(str(repeat), triangulate=False), 0)
        shift_ascii_ply(str(repeat), self.center)
        self.assertEqual(repeat.read_bytes(), self.mesh_bytes)

    def run_entry(self, script, out, *options, npz=None, expected_code=0):
        # 仅限制测试子进程的线程数，不修改 GUI/父进程环境。
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(ROOT / "windows_compat" / script),
             "--npz", str(self.npz if npz is None else npz),
             "--center-json", str(self.center_path),
             "--out", str(out), *options],
            cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        return result

    def test_new_entry_matches_old_geometry_and_keeps_inputs_read_only(self):
        source_hash = sha256_file(self.npz)
        old_out = self.root / "old_entry" / "surface.ply"
        new_out = self.root / "new_entry" / "surface.ply"
        self.run_entry("run_psdr_compod.py", old_out, "--no-simplify")
        input_directory_hashes = {
            path.name: sha256_file(path) for path in self.root.iterdir() if path.is_file()
        }
        self.run_entry("run_psdr_compod_provenance.py", new_out, "--no-simplify")
        self.assertEqual(old_out.read_bytes(), new_out.read_bytes())
        self.assertEqual(sha256_file(self.npz), source_hash)
        # 完整图割结果不要求等于解析 occupancy；只比较新旧入口的真实网格。
        metadata = load_surface_provenance(new_out)
        self.assertGreater(metadata["face_count"], 0)
        self.assertEqual(len(metadata["face_plane_ids"]), metadata["face_count"])
        sanitized = new_out.with_name("surface_compod_input.npz")
        self.assertTrue(sanitized.is_file())
        self.assertEqual(metadata["input_npz"]["sha256"], sha256_file(sanitized))
        self.assertEqual(input_directory_hashes, {
            path.name: sha256_file(path) for path in self.root.iterdir() if path.is_file()
        })

    def test_default_tree_simplification_matches_old_entry(self):
        old_out = self.root / "old_default" / "surface.ply"
        new_out = self.root / "new_default" / "surface.ply"
        self.run_entry("run_psdr_compod.py", old_out, "--save-cells")
        self.run_entry("run_psdr_compod_provenance.py", new_out, "--save-cells")
        self.assertEqual(old_out.read_bytes(), new_out.read_bytes())
        self.assertEqual(
            (old_out.parent / "volume_mesh.ply").read_bytes(),
            new_out.with_name("surface_volume_mesh.ply").read_bytes())
        self.assertGreater(load_surface_provenance(new_out)["face_count"], 0)

    def test_keep_empty_classes_uses_original_input_without_a_copy(self):
        with np.load(self.npz, allow_pickle=False) as archive:
            data = dict(archive)
        data["classes"] = np.zeros(0, dtype=np.int32)
        source = self.root / "cube_empty_classes.npz"
        np.savez(source, **data)
        out = self.root / "new_keep_empty_classes" / "surface.ply"
        source_hash = sha256_file(source)
        self.run_entry("run_psdr_compod_provenance.py", out, "--keep-classes", npz=source)
        self.assertEqual(sha256_file(source), source_hash)
        self.assertEqual(load_surface_provenance(out)["input_npz"]["sha256"], source_hash)
        self.assertFalse(out.with_name("surface_compod_input.npz").exists())

    def test_keep_nonempty_classes_reproduces_known_legacy_failure(self):
        # 记录旧库已存在的初始化缺陷；不在本次谱系改造中改变标签算法或旧入口。
        for script, directory in (
                ("run_psdr_compod.py", "old_keep_nonempty"),
                ("run_psdr_compod_provenance.py", "new_keep_nonempty")):
            out = self.root / directory / "surface.ply"
            result = self.run_entry(script, out, "--keep-classes", expected_code=1)
            self.assertIn("point_class_weights", result.stderr)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix(".provenance.json").exists())


class EntrySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compod_provenance_entry_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.npz, self.center_path, _ = make_fixture(self.root)
        self.out = self.root / "output" / "surface.ply"
        self.source_hash = sha256_file(self.npz)

    def tearDown(self):
        self.assertEqual(sha256_file(self.npz), self.source_hash)

    def arguments(self, *options):
        return ["--npz", str(self.npz), "--center-json", str(self.center_path),
                "--out", str(self.out), *options]

    def assert_refused(self, *options):
        with patch.object(provenance_entry, "VertexGroup") as reconstruct:
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    provenance_entry.main(self.arguments(*options))
            self.assertEqual(caught.exception.code, 2)
            reconstruct.assert_not_called()

    def test_existing_surface_is_not_overwritten(self):
        self.out.parent.mkdir()
        self.out.write_bytes(b"existing surface")
        self.assert_refused()
        self.assertEqual(self.out.read_bytes(), b"existing surface")
        self.assertFalse(self.out.with_suffix(".provenance.json").exists())

    def test_existing_metadata_is_not_overwritten(self):
        self.out.parent.mkdir()
        metadata = self.out.with_suffix(".provenance.json")
        metadata.write_bytes(b"existing metadata")
        self.assert_refused()
        self.assertEqual(metadata.read_bytes(), b"existing metadata")
        self.assertFalse(self.out.exists())

    def test_inputs_cannot_be_used_as_metadata(self):
        self.assert_refused("--provenance-out", str(self.npz))
        self.assertFalse(self.out.parent.exists())

    def test_same_output_and_metadata_are_rejected(self):
        self.assert_refused("--provenance-out", str(self.out))
        self.assertFalse(self.out.parent.exists())

    def test_non_ply_output_is_rejected(self):
        self.assert_refused("--out", str(self.out.with_suffix(".off")))
        self.assertFalse(self.out.parent.exists())

    def test_existing_sanitized_copy_is_not_overwritten(self):
        self.out.parent.mkdir()
        sanitized = self.out.with_name("surface_compod_input.npz")
        sanitized.write_bytes(b"existing sanitized input")
        self.assert_refused()
        self.assertEqual(sanitized.read_bytes(), b"existing sanitized input")
        self.assertFalse(self.out.exists())

    def test_existing_translation_temporary_file_is_not_overwritten(self):
        self.out.parent.mkdir()
        temporary = Path(str(self.out) + ".tmp")
        temporary.write_bytes(b"existing translation temporary")
        self.assert_refused()
        self.assertEqual(temporary.read_bytes(), b"existing translation temporary")
        self.assertFalse(self.out.exists())

    def test_metadata_cannot_reuse_translation_temporary_path(self):
        self.assert_refused("--provenance-out", str(self.out) + ".tmp")
        self.assertFalse(self.out.parent.exists())

    def test_invalid_center_is_rejected_before_creating_outputs(self):
        self.center_path.write_text(json.dumps({"center": [float("nan"), 0, 0]}),
                                    encoding="utf-8")
        with patch.object(provenance_entry, "VertexGroup") as reconstruct:
            with self.assertRaisesRegex(ValueError, "finite"):
                provenance_entry.main(self.arguments())
            reconstruct.assert_not_called()
        self.assertFalse(self.out.parent.exists())


if __name__ == "__main__":
    unittest.main()
