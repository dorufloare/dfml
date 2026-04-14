#pragma once

#include "dfml/tensor.hpp"

namespace dfml {

inline float binary_accuracy(const Tensor<float>& pred, const Tensor<float>& target) {
    if (pred.nr_elements() != target.nr_elements())
        throw std::invalid_argument("accuracy: shape mismatch");

    const float* p = pred.data();
    const float* t = target.data();
    size_t n = pred.nr_elements();
    size_t correct = 0;

    for (size_t i = 0; i < n; ++i)
        if ((p[i] > 0.5f) == (t[i] > 0.5f)) ++correct;

    return static_cast<float>(correct) / static_cast<float>(n);
}

inline float mse(const Tensor<float>& pred, const Tensor<float>& target) {
    if (pred.nr_elements() != target.nr_elements())
        throw std::invalid_argument("mse: shape mismatch");

    const float* p = pred.data();
    const float* t = target.data();
    size_t n = pred.nr_elements();
    float sum = 0.f;

    for (size_t i = 0; i < n; ++i) {
        float diff = p[i] - t[i];
        sum += diff * diff;
    }

    return sum / static_cast<float>(n);
}

inline float mae(const Tensor<float>& pred, const Tensor<float>& target) {
    if (pred.nr_elements() != target.nr_elements())
        throw std::invalid_argument("mae: shape mismatch");

    const float* p = pred.data();
    const float* t = target.data();
    size_t n = pred.nr_elements();
    float sum = 0.f;

    for (size_t i = 0; i < n; ++i)
        sum += std::abs(p[i] - t[i]);

    return sum / static_cast<float>(n);
}


} //namespace dfml