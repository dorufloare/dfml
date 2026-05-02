#pragma once

#include "dfml/autograd/tensor_autograd.hpp"
#include "dfml/autograd/grad_guard.hpp"
#include "dfml/tensor.hpp"

#include <immintrin.h>

namespace dfml::ops {

inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template<typename T>
Tensor<T> matrix_multiply_avx2(const Tensor<T>& a, const Tensor<T>& b, bool parallelize = true) {
    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = b.size(1);

    const bool require_grad = GradGuard::is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    Tensor<T> c({M, N}, require_grad);
    const T* a_ptr = a.data();
    const T* b_ptr = b.data();
    T* c_ptr = c.data();

    // c = a * b
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            T a_val = a_ptr[i * K + k];

            if constexpr (std::is_same_v<T, float>) {
                __m256 va = _mm256_set1_ps(a_val);
                size_t j = 0;
                for (; j + 8 <= N; j += 8) {
                    __m256 vb = _mm256_loadu_ps(b_ptr + k * N + j);
                    __m256 vc = _mm256_loadu_ps(c_ptr + i * N + j);
                    vc = _mm256_fmadd_ps(va, vb, vc);
                    _mm256_storeu_ps(c_ptr + i * N + j, vc);
                }
                for (; j < N; ++j)
                    c_ptr[i * N + j] += a_val * b_ptr[k * N + j];

            } else {
                for (size_t j = 0; j < N; ++j) {
                    c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
                }
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

                if constexpr (std::is_same_v<T, float>) {
                    #pragma omp parallel for schedule(dynamic) if(parallelize)
                    for (size_t i = 0; i < M; ++i) {
                        const float* dcrow = dc_ptr + i * N;
                        for (size_t k = 0; k < K; ++k) {
                            const float* brow = b_graph_ptr + k * N;
                            __m256 acc = _mm256_setzero_ps();
                            size_t j = 0;
                            for (; j + 8 <= N; j += 8)
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dcrow + j), _mm256_loadu_ps(brow + j), acc);
                            float r = hsum256_ps(acc);
                            for (; j < N; ++j)
                                r += dcrow[j] * brow[j];
                            dA_ptr[i * K + k] += r;
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(dynamic) if(parallelize)
                    for (size_t i = 0; i < M; ++i)
                        for (size_t k = 0; k < K; ++k)
                            for (size_t j = 0; j < N; ++j)
                                dA_ptr[i * K + k] += dc_ptr[i * N + j] * b_graph_ptr[k * N + j];
                }
                a_graph.accumulate_grad(dA);
            }

            if (b_graph.requires_grad()) {
                T* dB_ptr = dB.data();

                if constexpr (std::is_same_v<T, float>) {
                    #pragma omp parallel for schedule(dynamic) if(parallelize)
                    for (size_t k = 0; k < K; ++k) {
                        float* dbrow = dB_ptr + k * N;
                        for (size_t i = 0; i < M; ++i) {
                            float a_val = a_graph_ptr[i * K + k];
                            __m256 va = _mm256_set1_ps(a_val);
                            const float* dcrow = dc_ptr + i * N;
                            size_t j = 0;
                            for (; j + 8 <= N; j += 8) {
                                __m256 vdb = _mm256_loadu_ps(dbrow + j);
                                vdb = _mm256_fmadd_ps(va, _mm256_loadu_ps(dcrow + j), vdb);
                                _mm256_storeu_ps(dbrow + j, vdb);
                            }
                            for (; j < N; ++j)
                                dbrow[j] += a_val * dcrow[j];
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(dynamic) if(parallelize)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t i = 0; i < M; ++i) {
                            T a_val = a_graph_ptr[i * K + k];
                            for (size_t j = 0; j < N; ++j)
                                dB_ptr[k * N + j] += a_val * dc_ptr[i * N + j];
                        }
                }
                b_graph.accumulate_grad(dB);
            }
        });

    }

    return c;
}

} // namespace dfml::ops
