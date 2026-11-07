cmake_minimum_required(VERSION 3.20)
project(psdr_cpp LANGUAGES CXX)

# Standalone C++ build of the official PSDR core (no pypsdr / nanobind / Python).
# Configure with the vcpkg toolchain, triplet x64-windows:
#   cmake -S . -B build -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=D:/github/vcpkg/scripts/buildsystems/vcpkg.cmake \
#     -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_BUILD_TYPE=Release

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CGAL_NO_DEPRECATED_CODE must stay OFF: PSDR uses the CGAL 5 aliases
# CGAL::Color / CGAL::black(), which CGAL 6 keeps only in its deprecated block.
# This vcpkg tree has fmt.dll but no lib/fmt.lib, and spdlog ships without
# its bundled fmt.  Header-only spdlog + header-only fmt avoids both binaries.
add_compile_definitions(_HAS_STD_BYTE=0 NOMINMAX CGAL_NO_GMP=1 CGAL_DISABLE_GMP=1
                        FMT_HEADER_ONLY)
if(MSVC)
    add_compile_options(/bigobj /EHsc /utf-8 /MP)
endif()

# vcpkg's BoostConfig cannot resolve every component here (and
# boost_program_options is not installed), so link the import libraries
# directly.  MSVC Boost headers auto-link as well; these make it explicit.
set(VCPKG_LIB "D:/github/vcpkg/installed/x64-windows/lib")
set(BOOST_LIBS
    "${VCPKG_LIB}/boost_filesystem-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_system-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_thread-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_serialization-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_chrono-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_date_time-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/boost_atomic-vc143-mt-x64-1_86.lib"
    "${VCPKG_LIB}/zlib.lib")

# CGAL 6 is header-only.  vcpkg's CGALConfig.cmake mis-detects CGAL_ROOT in
# this installation, so the headers are referenced directly instead.  GMP is
# disabled via CGAL_NO_GMP (PSDR uses the inexact EPICK kernel) to avoid
# needing an MSVC import library for gmp.dll.
# xtl / xtensor / xtensor-io are header-only too and are vendored below;
# vcpkg cannot build their ports here (HighFive pull-in / SDK rc.exe).
set(PSDR_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
set(XTENSOR_HEADERS "${PSDR_VENDOR}/thirdparty/include")
set(VCPKG_INC "D:/github/vcpkg/installed/x64-windows/include")

add_library(ShapeDetection STATIC
    "${PSDR_VENDOR}/src/shape_container.cpp"
    "${PSDR_VENDOR}/src/shape_detector.cpp"
    "${PSDR_VENDOR}/src/shape_detector_index_map.cpp"
    "${PSDR_VENDOR}/src/MeanShift.cpp"
)
target_include_directories(ShapeDetection PUBLIC
    "${PSDR_VENDOR}/include" "${XTENSOR_HEADERS}"
    "${VCPKG_INC}" "${VCPKG_INC}/eigen3")
target_link_libraries(ShapeDetection PUBLIC ${BOOST_LIBS})

add_executable(psdr_cli psdr_cli.cpp)
target_link_libraries(psdr_cli PRIVATE ShapeDetection)
