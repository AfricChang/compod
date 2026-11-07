#!/usr/bin/env bash
# End-to-end: OFF -> PSDR C++ (detect + refine) -> npz -> COMPOD -> surface PLY
# Usage: bash windows_compat/psdr_cpp/run_pipeline.sh [path/to.off]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/windows_compat/.venv/Scripts/python.exe"
CLI="$HERE/build/psdr_cli.exe"
OFF="${1:-$ROOT/data/oblique/anhui/smid_004.off}"
TAG="$(basename "$OFF" .off)"
WORK="$ROOT/data/oblique/result/psdr_cpp/$TAG"

export PATH="/d/github/vcpkg/installed/x64-windows/bin:/d/Windows Kits/10/bin/10.0.19041.0/x64:$PATH"
mkdir -p "$WORK"

echo "[1/3] OFF -> point cloud PLY (centred)"
"$PY" "$ROOT/windows_compat/off_to_pointcloud.py" "$OFF" "$WORK/pointcloud.ply" \
    --samples 200000 --center-json "$WORK/center.json"

echo "[2/3] PSDR C++ detect + refine -> npz"
"$CLI" "$WORK/pointcloud.ply" --out "$WORK/psdr_refined.npz" \
    --detected "$WORK/psdr_detected.npz" \
    --epsilon-ratio 0.01 --min-inliers 120 --knn 10 --normal-th 0.9

echo "[3/3] npz -> COMPOD -> surface PLY (coordinates restored)"
"$PY" "$ROOT/windows_compat/run_psdr_compod.py" \
    --npz "$WORK/psdr_refined.npz" --center-json "$WORK/center.json" \
    --out "$WORK/${TAG}_psdr_surface.ply"

echo "done: $WORK/${TAG}_psdr_surface.ply"
