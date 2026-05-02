#pragma once

#include "dfml/ops/matrix_multiply/naive.hpp"
#include "dfml/ops/matrix_multiply/simd.hpp"
#include "dfml/ops/matrix_multiply/tiled.hpp"

namespace dfml::ops {

template<typename T>
Tensor<T> matrix_multiply(const Tensor<T>& a, const Tensor<T>& b, bool can_parallelize = true, bool force_parallelize = false) {
    if (a.nr_dimensions() != 2 || b.nr_dimensions() != 2)
        throw std::invalid_argument("matrix_multiply: inputs must be 2D");
    if (a.size(1) != b.size(0))
        throw std::invalid_argument("matrix_multiply: input dimensions must match");

    const size_t N = b.size(1);
    constexpr size_t PARALLEL_THRESHOLD = 512;  // parallel wins at N >= 512 per benchmarks

    const bool parallelize = force_parallelize || (can_parallelize && N >= PARALLEL_THRESHOLD);

    if (N >= 1024)
        return matrix_multiply_avx2_tiled(a, b, parallelize);
    else if (N >= 128)
        return matrix_multiply_avx2(a, b, parallelize);
    else
        return matrix_multiply_naive(a, b, parallelize);
}

} // namespace dfml::ops
