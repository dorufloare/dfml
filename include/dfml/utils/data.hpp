#pragma once

#include "dfml/tensor.hpp"
#include "dfml/utils/random.hpp"

namespace dfml {

template<typename T>
std::pair<Tensor<T>, Tensor<T>> train_test_split(const Tensor<T>& X, float ratio) {
    size_t n = X.size(0);
    size_t split_index = static_cast<size_t>(static_cast<float>(n) * ratio);
    size_t row_stride = X.nr_elements() / n;

    Tensor<T> train({split_index, X.size(1)});
    Tensor<T> test({X.size(0) - split_index, X.size(1)});

    std::copy(X.data(), X.data() + split_index * row_stride, train.data());
    std::copy(X.data() + split_index * row_stride, X.data() + X.nr_elements(), test.data());

    return {train, test};
}

template<typename T>
void shuffle(Tensor<T>& X, Tensor<T>& Y) {
    std::mt19937& rng = global_rng();

    size_t n = X.size(0);
    size_t x_stride = X.nr_elements() / n;
    size_t y_stride = Y.nr_elements() / n;

    for (size_t i = n - 1; i > 0; --i) {
        size_t j = std::uniform_int_distribution<size_t>(0, i)(rng);

        std::swap_ranges(X.data() + i * x_stride, X.data() + i * x_stride + x_stride, X.data() + j * x_stride);
        std::swap_ranges(Y.data() + i * y_stride, Y.data() + i * y_stride + y_stride, Y.data() + j * y_stride);
    }
}   

}
