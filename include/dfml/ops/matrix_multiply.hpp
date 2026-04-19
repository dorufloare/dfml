#pragma once

#include "dfml/autograd/tensor_autograd.hpp"
#include "dfml/autograd/grad_guard.hpp"
#include "dfml/tensor.hpp"

#include <immintrin.h>

namespace dfml::ops {

constexpr size_t TILE_K = 256;  // large K -> B column stays in L1
constexpr size_t TILE_M = 32;   // small M -> A panel (32*256=8KB) fits in L2
constexpr size_t TILE_N = 128;  // wide N -> B panel (256*128=128KB) fits in L3
constexpr size_t MR = 4;        // rows processed simultaneously in the micro-kernel

__attribute__((always_inline)) void inner_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    size_t K, size_t N,
    size_t i0, size_t k0, size_t j0,
    size_t i_end, size_t k_end, size_t j_end)
{
    size_t i = i0;

    for (; i + MR <= i_end; i += MR) {
        for (size_t k = k0; k < k_end; ++k) {
            __m256 va0 = _mm256_set1_ps(a[i       * K + k]);
            __m256 va1 = _mm256_set1_ps(a[(i + 1) * K + k]);
            __m256 va2 = _mm256_set1_ps(a[(i + 2) * K + k]);
            __m256 va3 = _mm256_set1_ps(a[(i + 3) * K + k]);

            const float* brow = b + k * N;
            float* crow0 = c + i       * N;
            float* crow1 = c + (i + 1) * N;
            float* crow2 = c + (i + 2) * N;
            float* crow3 = c + (i + 3) * N;

            size_t j = j0;
            for (; j + 16 <= j_end; j += 16) {
                __m256 vb0 = _mm256_loadu_ps(brow + j);
                __m256 vb1 = _mm256_loadu_ps(brow + j + 8);
                __m256 vc00 = _mm256_loadu_ps(crow0 + j);      __m256 vc01 = _mm256_loadu_ps(crow0 + j + 8);
                __m256 vc10 = _mm256_loadu_ps(crow1 + j);      __m256 vc11 = _mm256_loadu_ps(crow1 + j + 8);
                __m256 vc20 = _mm256_loadu_ps(crow2 + j);      __m256 vc21 = _mm256_loadu_ps(crow2 + j + 8);
                __m256 vc30 = _mm256_loadu_ps(crow3 + j);      __m256 vc31 = _mm256_loadu_ps(crow3 + j + 8);
                vc00 = _mm256_fmadd_ps(va0, vb0, vc00);        vc01 = _mm256_fmadd_ps(va0, vb1, vc01);
                vc10 = _mm256_fmadd_ps(va1, vb0, vc10);        vc11 = _mm256_fmadd_ps(va1, vb1, vc11);
                vc20 = _mm256_fmadd_ps(va2, vb0, vc20);        vc21 = _mm256_fmadd_ps(va2, vb1, vc21);
                vc30 = _mm256_fmadd_ps(va3, vb0, vc30);        vc31 = _mm256_fmadd_ps(va3, vb1, vc31);
                _mm256_storeu_ps(crow0 + j,     vc00);         _mm256_storeu_ps(crow0 + j + 8, vc01);
                _mm256_storeu_ps(crow1 + j,     vc10);         _mm256_storeu_ps(crow1 + j + 8, vc11);
                _mm256_storeu_ps(crow2 + j,     vc20);         _mm256_storeu_ps(crow2 + j + 8, vc21);
                _mm256_storeu_ps(crow3 + j,     vc30);         _mm256_storeu_ps(crow3 + j + 8, vc31);
            }
           
            for (; j + 8 <= j_end; j += 8) {
                __m256 vb = _mm256_loadu_ps(brow + j);
                __m256 vc0 = _mm256_loadu_ps(crow0 + j); vc0 = _mm256_fmadd_ps(va0, vb, vc0); _mm256_storeu_ps(crow0 + j, vc0);
                __m256 vc1 = _mm256_loadu_ps(crow1 + j); vc1 = _mm256_fmadd_ps(va1, vb, vc1); _mm256_storeu_ps(crow1 + j, vc1);
                __m256 vc2 = _mm256_loadu_ps(crow2 + j); vc2 = _mm256_fmadd_ps(va2, vb, vc2); _mm256_storeu_ps(crow2 + j, vc2);
                __m256 vc3 = _mm256_loadu_ps(crow3 + j); vc3 = _mm256_fmadd_ps(va3, vb, vc3); _mm256_storeu_ps(crow3 + j, vc3);
            }
           
            float a0 = a[i * K + k], a1 = a[(i+1)*K+k], a2 = a[(i+2)*K+k], a3 = a[(i+3)*K+k];
            for (; j < j_end; ++j) {
                float bv = brow[j];
                crow0[j] += a0 * bv;
                crow1[j] += a1 * bv;
                crow2[j] += a2 * bv;
                crow3[j] += a3 * bv;
            }
        }
    }


    for (; i < i_end; ++i) {
        for (size_t k = k0; k < k_end; ++k) {
            float a_val = a[i * K + k];
            __m256 va = _mm256_set1_ps(a_val);
            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 vb = _mm256_loadu_ps(b + k * N + j);
                __m256 vc = _mm256_loadu_ps(c + i * N + j);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(c + i * N + j, vc);
            }
            for (; j < j_end; ++j)
                c[i * N + j] += a_val * b[k * N + j];
        }
    }
}

template<typename T>
Tensor<T> matrix_multiply_avx2_tiled(const Tensor<T>& a, const Tensor<T>& b, bool parallelize = true) {
    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = b.size(1);

    const bool require_grad = GradGuard::is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    Tensor<T> c({M, N}, require_grad);
    const T* a_ptr = a.data();
    const T* b_ptr = b.data();
    T* c_ptr = c.data();

    if constexpr (std::is_same_v<T, float>) {
        for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
            size_t k_end = std::min(k0 + TILE_K, K);
            #pragma omp parallel for collapse(2) schedule(dynamic) if(parallelize)
            for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                    size_t i_end = std::min(i0 + TILE_M, M);
                    size_t j_end = std::min(j0 + TILE_N, N);
                    inner_kernel(a_ptr, b_ptr, c_ptr, K, N, i0, k0, j0, i_end, k_end, j_end);
                }
            }
        }
    } else {
        for (size_t i = 0; i < M; ++i)
            for (size_t k = 0; k < K; ++k) {
                T a_val = a_ptr[i * K + k];
                for (size_t j = 0; j < N; ++j)
                    c_ptr[i * N + j] += a_val * b_ptr[k * N + j];
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

        // weak_ptr<TensorImpl> instead of Tensor to solve circularity
        const auto c_weak = c.make_weak_tensor();

        c.set_backward_function([a_graph, b_graph, c_weak, M, K, N, parallelize]() mutable {
            auto c_locked = Tensor<T>::lock_weak_tensor(c_weak);
            if (!c_locked.has_value()) return;

            const T* dc_ptr = c_locked->grad().data();
            const T* a_graph_ptr = a_graph.data();
            const T* b_graph_ptr = b_graph.data();

            if (a_graph.requires_grad()) {
                Tensor<T> dA({M, K});
                dA.zero();
                T* dA_ptr = dA.data();
                // safe to parallelize over i: each i writes to a distinct row of dA
                #pragma omp parallel for schedule(dynamic) if(parallelize)
                for (size_t i = 0; i < M; ++i)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t j = 0; j < N; ++j)
                            dA_ptr[i * K + k] += dc_ptr[i * N + j] * b_graph_ptr[k * N + j];
                a_graph.accumulate_grad(dA);
            }

            if (b_graph.requires_grad()) {
                Tensor<T> dB({K, N});
                dB.zero();
                T* dB_ptr = dB.data();
                // parallelize over k: each k writes to a distinct row of dB
                // (cannot parallelize over i bcs multiple i values accumulate into the same dB[k][j])
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

        // weak_ptr<TensorImpl> instead of Tensor to solve circularity
        const auto c_weak = c.make_weak_tensor();

        c.set_backward_function([a_graph, b_graph, c_weak, M, K, N, parallelize]() mutable {
            auto c_locked = Tensor<T>::lock_weak_tensor(c_weak);
            if (!c_locked.has_value()) return;

            const T* dc_ptr = c_locked->grad().data();
            const T* a_graph_ptr = a_graph.data();
            const T* b_graph_ptr = b_graph.data();

            if (a_graph.requires_grad()) {
                Tensor<T> dA({M, K});
                dA.zero();
                T* dA_ptr = dA.data();
           
                #pragma omp parallel for schedule(dynamic) if(parallelize)
                for (size_t i = 0; i < M; ++i)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t j = 0; j < N; ++j)
                            dA_ptr[i * K + k] += dc_ptr[i * N + j] * b_graph_ptr[k * N + j];
                a_graph.accumulate_grad(dA);
            }

            if (b_graph.requires_grad()) {
                Tensor<T> dB({K, N});
                dB.zero();
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

        // weak_ptr<TensorImpl> instead of Tensor to solve circularity
        const auto c_weak = c.make_weak_tensor();

        c.set_backward_function([a_graph, b_graph, c_weak, M, K, N, parallelize]() mutable {
            auto c_locked = Tensor<T>::lock_weak_tensor(c_weak);
            if (!c_locked.has_value()) return;

            const T* dc_ptr = c_locked->grad().data();
            const T* a_graph_ptr = a_graph.data();
            const T* b_graph_ptr = b_graph.data();

            if (a_graph.requires_grad()) {
                Tensor<T> dA({M, K});
                dA.zero();
                T* dA_ptr = dA.data();
               
                #pragma omp parallel for schedule(dynamic) if(parallelize)
                for (size_t i = 0; i < M; ++i)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t j = 0; j < N; ++j)
                            dA_ptr[i * K + k] += dc_ptr[i * N + j] * b_graph_ptr[k * N + j];
                a_graph.accumulate_grad(dA);
            }

            if (b_graph.requires_grad()) {
                Tensor<T> dB({K, N});
                dB.zero();
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

} //namespace dfml::ops