#pragma once

#include "dfml/autograd/tensor_autograd.hpp"
#include "dfml/autograd/grad_guard.hpp"
#include "dfml/tensor.hpp"

namespace dfml::ops {

template<typename T>
Tensor<T> matrix_multiply_naive(const Tensor<T>& a, const Tensor<T>& b, bool parallelize = true) {
    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = b.size(1);

    const bool require_grad = GradGuard::is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    Tensor<T> c({M, N}, require_grad);
    const T* a_ptr = a.data();
    const T* b_ptr = b.data();
    T* c_ptr = c.data();

    // c = a * b
    #pragma omp parallel for schedule(dynamic) if(parallelize)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            T a_val = a_ptr[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
            }
        }
    }

    if (require_grad) {

        // during backward we know dL / dC
        // compute dL / dA
        // a[i][k] appears in every c[i][j] for all j
        // c[i][j] = ... + a[i][k] * b[k][j] + ...

        // dL / dA = dL/dC * dC/dA
        // => dL / dA[i][k] = sum(dL / dC[i][j] * B[k][j], j=0..N)
        // => dL / dA = dL / dC * B_t

        Tensor<T> a_graph = a;
        Tensor<T> b_graph = b;

        c.set_previous_tensors({a_graph, b_graph});

        c.add_grad_preallocated_workspace({M, K});
        c.add_grad_preallocated_workspace({K, N});    

        // weak_ptr<TensorImpl> instead of Tensor to solve circularity
        const auto c_weak = c.make_weak_tensor();

        c.set_backward_function([a_graph, b_graph, c_weak, M, K, N, parallelize]() mutable {
            auto c_locked = Tensor<T>::lock_weak_tensor(c_weak);
            if (!c_locked.has_value()) return;

            const T* dc_ptr = c_locked->grad().data();
            const T* a_graph_ptr = a_graph.data();
            const T* b_graph_ptr = b_graph.data();

            Tensor<T> dA = c_locked->get_grad_preallocated_workspace(0);
            Tensor<T> dB = c_locked->get_grad_preallocated_workspace(1);
            dA.zero();   
            dB.zero();

            if (a_graph.requires_grad()) {
                T* dA_ptr = dA.data();

                #pragma omp parallel for schedule(dynamic) if(parallelize)
                for (size_t i = 0; i < M; ++i)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t j = 0; j < N; ++j)
                            dA_ptr[i * K + k] += dc_ptr[i * N + j] * b_graph_ptr[k * N + j];
                a_graph.accumulate_grad(dA);
            }

            if (b_graph.requires_grad()) {
                T* dB_ptr = dB.data();

                #pragma omp parallel for schedule(dynamic) if(parallelize)
                for (size_t k = 0; k < K; ++k)
                    for (size_t i = 0; i < M; ++i) {
                        T a_val = a_graph_ptr[i * K + k];
                        for (size_t j = 0; j < N; ++j)
                            dB_ptr[k * N + j] += a_val * dc_ptr[i * N + j];
                    }
                b_graph.accumulate_grad(dB);
            }
        });

    }

    return c;
}

} // namespace dfml::ops
