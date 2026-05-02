#pragma once

#include "dfml/ops/matrix_multiply/simd.hpp"  // hsum256_ps + AVX2 includes

namespace dfml::ops {

constexpr size_t TILE_K = 256;  // large K -> B column stays in L1
constexpr size_t TILE_M = 32;   // small M -> A panel (32*256=8KB) fits in L2
constexpr size_t TILE_N = 128;  // wide N -> B panel (256*128=128KB) fits in L3
constexpr size_t MR = 4;        // rows processed simultaneously in the micro-kernel


__attribute__((always_inline)) void inner_kernel_dB(
    const float* __restrict__ a,
    const float* __restrict__ dc,
    float* __restrict__ db,
    size_t K, size_t N,
    size_t k0, size_t i0, size_t j0,
    size_t k_end, size_t i_end, size_t j_end)
{
    size_t k = k0;

    for (; k + MR <= k_end; k += MR) {
        for (size_t i = i0; i < i_end; ++i) {
            __m256 va0 = _mm256_set1_ps(a[i * K + k]);
            __m256 va1 = _mm256_set1_ps(a[i * K + k + 1]);
            __m256 va2 = _mm256_set1_ps(a[i * K + k + 2]);
            __m256 va3 = _mm256_set1_ps(a[i * K + k + 3]);

            const float* dcrow = dc + i * N;
            float* dbrow0 = db + k       * N;
            float* dbrow1 = db + (k + 1) * N;
            float* dbrow2 = db + (k + 2) * N;
            float* dbrow3 = db + (k + 3) * N;

            size_t j = j0;
            for (; j + 16 <= j_end; j += 16) {
                __m256 vd0 = _mm256_loadu_ps(dcrow + j);
                __m256 vd1 = _mm256_loadu_ps(dcrow + j + 8);
                __m256 vb00 = _mm256_loadu_ps(dbrow0 + j);     __m256 vb01 = _mm256_loadu_ps(dbrow0 + j + 8);
                __m256 vb10 = _mm256_loadu_ps(dbrow1 + j);     __m256 vb11 = _mm256_loadu_ps(dbrow1 + j + 8);
                __m256 vb20 = _mm256_loadu_ps(dbrow2 + j);     __m256 vb21 = _mm256_loadu_ps(dbrow2 + j + 8);
                __m256 vb30 = _mm256_loadu_ps(dbrow3 + j);     __m256 vb31 = _mm256_loadu_ps(dbrow3 + j + 8);
                vb00 = _mm256_fmadd_ps(va0, vd0, vb00);        vb01 = _mm256_fmadd_ps(va0, vd1, vb01);
                vb10 = _mm256_fmadd_ps(va1, vd0, vb10);        vb11 = _mm256_fmadd_ps(va1, vd1, vb11);
                vb20 = _mm256_fmadd_ps(va2, vd0, vb20);        vb21 = _mm256_fmadd_ps(va2, vd1, vb21);
                vb30 = _mm256_fmadd_ps(va3, vd0, vb30);        vb31 = _mm256_fmadd_ps(va3, vd1, vb31);
                _mm256_storeu_ps(dbrow0 + j,     vb00);        _mm256_storeu_ps(dbrow0 + j + 8, vb01);
                _mm256_storeu_ps(dbrow1 + j,     vb10);        _mm256_storeu_ps(dbrow1 + j + 8, vb11);
                _mm256_storeu_ps(dbrow2 + j,     vb20);        _mm256_storeu_ps(dbrow2 + j + 8, vb21);
                _mm256_storeu_ps(dbrow3 + j,     vb30);        _mm256_storeu_ps(dbrow3 + j + 8, vb31);
            }

            for (; j + 8 <= j_end; j += 8) {
                __m256 vd = _mm256_loadu_ps(dcrow + j);
                __m256 vb0 = _mm256_loadu_ps(dbrow0 + j); vb0 = _mm256_fmadd_ps(va0, vd, vb0); _mm256_storeu_ps(dbrow0 + j, vb0);
                __m256 vb1 = _mm256_loadu_ps(dbrow1 + j); vb1 = _mm256_fmadd_ps(va1, vd, vb1); _mm256_storeu_ps(dbrow1 + j, vb1);
                __m256 vb2 = _mm256_loadu_ps(dbrow2 + j); vb2 = _mm256_fmadd_ps(va2, vd, vb2); _mm256_storeu_ps(dbrow2 + j, vb2);
                __m256 vb3 = _mm256_loadu_ps(dbrow3 + j); vb3 = _mm256_fmadd_ps(va3, vd, vb3); _mm256_storeu_ps(dbrow3 + j, vb3);
            }

            float a0 = a[i*K + k], a1 = a[i*K + k + 1], a2 = a[i*K + k + 2], a3 = a[i*K + k + 3];
            for (; j < j_end; ++j) {
                float dv = dcrow[j];
                dbrow0[j] += a0 * dv;
                dbrow1[j] += a1 * dv;
                dbrow2[j] += a2 * dv;
                dbrow3[j] += a3 * dv;
            }

            if (i + 2 < i_end)
                _mm_prefetch(reinterpret_cast<const char*>(dc + (i + 2) * N + j0), _MM_HINT_T1);
        }
    }

    for (; k < k_end; ++k) {
        for (size_t i = i0; i < i_end; ++i) {
            float a_val = a[i * K + k];
            __m256 va = _mm256_set1_ps(a_val);
            const float* dcrow = dc + i * N;
            float* dbrow = db + k * N;
            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 vd = _mm256_loadu_ps(dcrow + j);
                __m256 vb = _mm256_loadu_ps(dbrow + j);
                vb = _mm256_fmadd_ps(va, vd, vb);
                _mm256_storeu_ps(dbrow + j, vb);
            }
            for (; j < j_end; ++j)
                dbrow[j] += a_val * dcrow[j];
        }
    }
}

__attribute__((always_inline)) void inner_kernel_dA(
    const float* __restrict__ dc,
    const float* __restrict__ b,
    float* __restrict__ da,
    size_t K, size_t N,
    size_t i0, size_t k0, size_t j0,
    size_t i_end, size_t k_end, size_t j_end)
{
    size_t i = i0;

    for (; i + MR <= i_end; i += MR) {
        size_t k = k0;
        const float* dc0 = dc + i       * N;
        const float* dc1 = dc + (i + 1) * N;
        const float* dc2 = dc + (i + 2) * N;
        const float* dc3 = dc + (i + 3) * N;

        for (; k + MR <= k_end; k += MR) {
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
            __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();
            __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps(), a32 = _mm256_setzero_ps(), a33 = _mm256_setzero_ps();

            const float* b0 = b + k       * N;
            const float* b1 = b + (k + 1) * N;
            const float* b2 = b + (k + 2) * N;
            const float* b3 = b + (k + 3) * N;

            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 vd0 = _mm256_loadu_ps(dc0 + j);
                __m256 vd1 = _mm256_loadu_ps(dc1 + j);
                __m256 vd2 = _mm256_loadu_ps(dc2 + j);
                __m256 vd3 = _mm256_loadu_ps(dc3 + j);
                __m256 vb0 = _mm256_loadu_ps(b0 + j);
                __m256 vb1 = _mm256_loadu_ps(b1 + j);
                __m256 vb2 = _mm256_loadu_ps(b2 + j);
                __m256 vb3 = _mm256_loadu_ps(b3 + j);
                a00 = _mm256_fmadd_ps(vd0, vb0, a00); a01 = _mm256_fmadd_ps(vd0, vb1, a01); a02 = _mm256_fmadd_ps(vd0, vb2, a02); a03 = _mm256_fmadd_ps(vd0, vb3, a03);
                a10 = _mm256_fmadd_ps(vd1, vb0, a10); a11 = _mm256_fmadd_ps(vd1, vb1, a11); a12 = _mm256_fmadd_ps(vd1, vb2, a12); a13 = _mm256_fmadd_ps(vd1, vb3, a13);
                a20 = _mm256_fmadd_ps(vd2, vb0, a20); a21 = _mm256_fmadd_ps(vd2, vb1, a21); a22 = _mm256_fmadd_ps(vd2, vb2, a22); a23 = _mm256_fmadd_ps(vd2, vb3, a23);
                a30 = _mm256_fmadd_ps(vd3, vb0, a30); a31 = _mm256_fmadd_ps(vd3, vb1, a31); a32 = _mm256_fmadd_ps(vd3, vb2, a32); a33 = _mm256_fmadd_ps(vd3, vb3, a33);
            }

            float r00 = hsum256_ps(a00), r01 = hsum256_ps(a01), r02 = hsum256_ps(a02), r03 = hsum256_ps(a03);
            float r10 = hsum256_ps(a10), r11 = hsum256_ps(a11), r12 = hsum256_ps(a12), r13 = hsum256_ps(a13);
            float r20 = hsum256_ps(a20), r21 = hsum256_ps(a21), r22 = hsum256_ps(a22), r23 = hsum256_ps(a23);
            float r30 = hsum256_ps(a30), r31 = hsum256_ps(a31), r32 = hsum256_ps(a32), r33 = hsum256_ps(a33);

            for (; j < j_end; ++j) {
                float d0 = dc0[j], d1 = dc1[j], d2 = dc2[j], d3 = dc3[j];
                float v0 = b0[j],  v1 = b1[j],  v2 = b2[j],  v3 = b3[j];
                r00 += d0*v0; r01 += d0*v1; r02 += d0*v2; r03 += d0*v3;
                r10 += d1*v0; r11 += d1*v1; r12 += d1*v2; r13 += d1*v3;
                r20 += d2*v0; r21 += d2*v1; r22 += d2*v2; r23 += d2*v3;
                r30 += d3*v0; r31 += d3*v1; r32 += d3*v2; r33 += d3*v3;
            }

            da[i*K + k]         += r00; da[i*K + k + 1]         += r01; da[i*K + k + 2]         += r02; da[i*K + k + 3]         += r03;
            da[(i + 1)*K + k]   += r10; da[(i + 1)*K + k + 1]   += r11; da[(i + 1)*K + k + 2]   += r12; da[(i + 1)*K + k + 3]   += r13;
            da[(i + 2)*K + k]   += r20; da[(i + 2)*K + k + 1]   += r21; da[(i + 2)*K + k + 2]   += r22; da[(i + 2)*K + k + 3]   += r23;
            da[(i + 3)*K + k]   += r30; da[(i + 3)*K + k + 1]   += r31; da[(i + 3)*K + k + 2]   += r32; da[(i + 3)*K + k + 3]   += r33;
        }

        for (; k < k_end; ++k) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            const float* brow = b + k * N;
            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                __m256 vb = _mm256_loadu_ps(brow + j);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(dc0 + j), vb, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(dc1 + j), vb, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(dc2 + j), vb, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(dc3 + j), vb, a3);
            }
            float r0 = hsum256_ps(a0), r1 = hsum256_ps(a1), r2 = hsum256_ps(a2), r3 = hsum256_ps(a3);
            for (; j < j_end; ++j) {
                float bv = brow[j];
                r0 += dc0[j] * bv;
                r1 += dc1[j] * bv;
                r2 += dc2[j] * bv;
                r3 += dc3[j] * bv;
            }
            da[i*K + k]         += r0;
            da[(i + 1)*K + k]   += r1;
            da[(i + 2)*K + k]   += r2;
            da[(i + 3)*K + k]   += r3;
        }
    }

    for (; i < i_end; ++i) {
        const float* dcrow = dc + i * N;
        for (size_t k = k0; k < k_end; ++k) {
            __m256 acc = _mm256_setzero_ps();
            const float* brow = b + k * N;
            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dcrow + j), _mm256_loadu_ps(brow + j), acc);
            }
            float r = hsum256_ps(acc);
            for (; j < j_end; ++j)
                r += dcrow[j] * brow[j];
            da[i*K + k] += r;
        }
    }
}

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

            if (k + 2 < k_end)
                _mm_prefetch(reinterpret_cast<const char*>(b + (k + 2) * N + j0), _MM_HINT_T1);
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
                    for (size_t j0 = 0; j0 < N; j0 += TILE_K) {
                        size_t j_end = std::min(j0 + TILE_K, N);
                        #pragma omp parallel for collapse(2) schedule(dynamic) if(parallelize)
                        for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
                            for (size_t k0 = 0; k0 < K; k0 += TILE_M) {
                                size_t i_end = std::min(i0 + TILE_M, M);
                                size_t k_end = std::min(k0 + TILE_M, K);
                                inner_kernel_dA(dc_ptr, b_graph_ptr, dA_ptr, K, N, i0, k0, j0, i_end, k_end, j_end);
                            }
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
                    for (size_t i0 = 0; i0 < M; i0 += TILE_K) {
                        size_t i_end = std::min(i0 + TILE_K, M);
                        #pragma omp parallel for collapse(2) schedule(dynamic) if(parallelize)
                        for (size_t k0 = 0; k0 < K; k0 += TILE_M) {
                            for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
                                size_t k_end = std::min(k0 + TILE_M, K);
                                size_t j_end = std::min(j0 + TILE_N, N);
                                inner_kernel_dB(a_graph_ptr, dc_ptr, dB_ptr, K, N, k0, i0, j0, k_end, i_end, j_end);
                            }
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
