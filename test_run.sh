#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
CONFIG="Release"

cmake -S . -B "$BUILD_DIR" -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE="$CONFIG"
cmake --build "$BUILD_DIR" --config "$CONFIG"

# CTest uses -C for multi-config generators (Visual Studio, Xcode).
if grep -q "^CMAKE_CONFIGURATION_TYPES:" "$BUILD_DIR/CMakeCache.txt"; then
  ctest --test-dir "$BUILD_DIR" -C "$CONFIG" --output-on-failure
else
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
