#!/bin/bash
# Azahar RK3568 optimized build script
# Targets: Cortex-A55 (ARMv8.2-A+DotProd+FP16), Mali-G52, 2-4GB RAM
set -e

BUILD_DIR="${1:-build-rk3568}"
NPROC=$(nproc)

echo "=== Azahar RK3568 Optimized Build ==="
echo "Build dir: ${BUILD_DIR}"
echo "Jobs: ${NPROC}"
echo ""

cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DAZAHAR_TARGET_CPU=cortex-a55 \
    -DENABLE_LTO=ON \
    -DENABLE_OPENGL=OFF \
    -DENABLE_VULKAN=ON \
    -DENABLE_SOFTWARE_RENDERER=OFF \
    -DENABLE_QT=ON \
    -DENABLE_TESTS=OFF \
    -DENABLE_ROOM=OFF \
    -DENABLE_ROOM_STANDALONE=OFF \
    -DENABLE_WEB_SERVICE=OFF \
    -DENABLE_SCRIPTING=OFF \
    -DENABLE_MICROPROFILE=OFF \
    -DENABLE_DEVELOPER_OPTIONS=OFF \
    -DENABLE_CUBEB=ON \
    -DENABLE_OPENAL=OFF \
    -DUSE_DISCORD_PRESENCE=OFF \
    -DCITRA_USE_PRECOMPILED_HEADERS=ON \
    -DCITRA_WARNINGS_AS_ERRORS=OFF \
    -DUSE_SYSTEM_SDL2=ON \
    -DUSE_SYSTEM_BOOST=ON \
    -DUSE_SYSTEM_FMT=ON \
    -DUSE_SYSTEM_ZSTD=ON \
    .

echo ""
echo "=== Building with ${NPROC} jobs ==="
cmake --build "${BUILD_DIR}" -j"${NPROC}"

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/bin/Release/azahar"
