#pragma once

#include "dfml/tensor.hpp"

#include <cmath>
#include <algorithm>

namespace dfml::ops {

//softmax[i][j] = exp(x[i][j] - max[i][...]) / sum(exp[i][j], j=0...)
//dL/dX[i][j] = softmax[i][j] * (dL/dR[i][j] - dot(dL/dR[i][...], softmax[i][...]))

template<typename T>
Tensor<T> softmax(const Tensor<T>& a) {
    if (a.nr_dimensions() != 2)
        throw std::invalid_argument("softmax: only 2D supported");

    const bool require_grad = GradGuard::is_grad_enabled() && a.requires_grad();
    Tensor<T> result(a.shape(), require_grad);

    const T* a_ptr = a.data();
    T* result_ptr = result.data();

    const size_t M = a.size(0);
    const size_t N = a.size(1);

    for (size_t i = 0; i < M; ++i) {
        const T* row_a = a_ptr + i * N;
        T* row_result = result_ptr + i * N;

        T row_max = *std::max_element(row_a, row_a + N);

        T sum = T{};
        for (size_t j = 0; j < N; ++j) {
            row_result[j] = std::exp(row_a[j] - row_max);
            sum += row_result[j];
        }

        for (size_t j = 0; j < N; ++j)
            row_result[j] /= sum;
    }

    if (require_grad) {
        Tensor<T> a_graph = a;

        result.set_previous_tensors({a_graph});

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([a_graph, N, M, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T* dr_ptr = result_locked->grad().data();
            const T* r_ptr = result_locked->data();

            Tensor<T> dA(a_graph.shape());
            T* da_ptr = dA.data();

            for (size_t i = 0; i < M; ++i) {
                const T* dr_row = dr_ptr + i * N;
                const T* r_row = r_ptr + i * N;
                T* da_row = da_ptr + i * N;

                T dot = T{};
                for (size_t j = 0; j < N; ++j) {
                    dot += dr_row[j] * r_row[j];
                }

                for (size_t j = 0; j < N; ++j) {
                    da_row[j] = r_row[j] * (dr_row[j] - dot);
                }
            }

            a_graph.accumulate_grad(dA);
        });

    }

    return result;
}

} //namespace dfml::ops