#pragma once

#include <cmath>
#include <random>

#include "dfml/tensor.hpp"

namespace dfml::init {

inline void xavier_uniform(Tensor<float>& t, size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);
    float* ptr = t.data();
    for (size_t i = 0; i < t.nr_elements(); ++i)
        ptr[i] = dist(rng);
}

inline void xavier_normal(Tensor<float>& t, size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.f / static_cast<float>(fan_in + fan_out));
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.f, std);
    float* ptr = t.data();
    for (size_t i = 0; i < t.nr_elements(); ++i)
        ptr[i] = dist(rng);
}

inline void kaiming_normal(Tensor<float>& t, size_t fan_in) {
    float std = std::sqrt(2.f / static_cast<float>(fan_in));
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.f, std);
    float* ptr = t.data();
    for (size_t i = 0; i < t.nr_elements(); ++i)
        ptr[i] = dist(rng);
}

inline void zeros(Tensor<float>& t) {
    t.zero();
}

} // namespace dfml::init
