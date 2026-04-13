#pragma once

#include "dfml/tensor.hpp"

namespace dfml::ops {

template<typename T>
Tensor<T> matrix_transpose(const Tensor<T>& a) {
    if (a.nr_dimensions() != 2) 
        throw std::invalid_argument("matrix_transpose: a must be a matrix (2 dimensions)");
    
    const size_t M = a.size(0);
    const size_t N = a.size(1);

    Tensor<T> result({N, M});

    const T* a_ptr = a.data();
    T* result_ptr = result.data();

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            result_ptr[j * M + i] = a_ptr[i * N + j];

    return result;
}

} //namespace dfml::ops