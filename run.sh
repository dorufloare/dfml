#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
APP_NAME="dfml"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release

# Handle both single-config and multi-config generators.
if [[ -x "$BUILD_DIR/$APP_NAME" ]]; then
  "$BUILD_DIR/$APP_NAME"
elif [[ -x "$BUILD_DIR/Release/$APP_NAME" ]]; then
  "$BUILD_DIR/Release/$APP_NAME"
elif [[ -x "$BUILD_DIR/$APP_NAME.exe" ]]; then
  "$BUILD_DIR/$APP_NAME.exe"
elif [[ -x "$BUILD_DIR/Release/$APP_NAME.exe" ]]; then
  "$BUILD_DIR/Release/$APP_NAME.exe"
else
  echo "Built executable not found for target '$APP_NAME'." >&2
  exit 1
fi
