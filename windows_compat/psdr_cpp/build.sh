#!/usr/bin/env bash
# Build the standalone PSDR C++ CLI with MSVC + vcpkg x64-windows.
# No Python / nanobind / scikit-build involved.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VCPKG="${VCPKG_ROOT:-D:/github/vcpkg}"
MSVC_VER="14.44.35207"
MSVC_POSIX="/d/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/$MSVC_VER"
MSVC_WIN='D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\'"$MSVC_VER"
SDK_WIN='D:\Windows Kits\10'
SDK_VER='10.0.26100.0'

# The VS CMake generator does not discover this D:-drive installation, so set
# up the MSVC environment by hand and use Ninja with cl.exe.
export INCLUDE="$MSVC_WIN\\include;$SDK_WIN\\Include\\$SDK_VER\\ucrt;$SDK_WIN\\Include\\$SDK_VER\\um;$SDK_WIN\\Include\\$SDK_VER\\shared;$SDK_WIN\\Include\\$SDK_VER\\winrt"
export LIB="$MSVC_WIN\\lib\\x64;$SDK_WIN\\Lib\\$SDK_VER\\ucrt\\x64;$SDK_WIN\\Lib\\$SDK_VER\\um\\x64"
export PATH="$MSVC_POSIX/bin/Hostx64/x64:/d/Windows Kits/10/bin/$SDK_VER/x64:$PATH"
CL="$MSVC_POSIX/bin/Hostx64/x64/cl.exe"

cmake -S "$HERE" -B "$HERE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CL" -DCMAKE_CXX_COMPILER="$CL" \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build "$HERE/build" --target psdr_cli -j 4
echo "built: $HERE/build/psdr_cli.exe"
