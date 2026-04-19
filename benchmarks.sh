#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build "$BUILD_DIR" --config Release --target bench_matrix_multiply

# handle single-config and multi-config generators
for path in \
    "$BUILD_DIR/bench_matrix_multiply" \
    "$BUILD_DIR/benchmarks/bench_matrix_multiply" \
    "$BUILD_DIR/Release/bench_matrix_multiply.exe" \
    "$BUILD_DIR/benchmarks/Release/bench_matrix_multiply.exe"
do
    [[ -x "$path" ]] && exec "$path"
done

echo "Benchmark executable not found." >&2
exit 1
