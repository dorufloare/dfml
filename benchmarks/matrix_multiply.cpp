#include "dfml/ops/matrix_multiply.hpp"
#include "dfml/tensor.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

namespace dfml {

static void fill_random(Tensor<float>& t) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    float* p = t.data();
    for (size_t i = 0; i < t.nr_elements(); ++i)
        p[i] = dist(rng);
}

template<typename MatmulFn>
static double bench(MatmulFn fn, size_t M, size_t K, size_t N, int reps = 5) {
    Tensor<float> a({M, K});
    Tensor<float> b({K, N});
    fill_random(a);
    fill_random(b);

    fn(a, b); // warm-up

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        fn(a, b);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

}

int main() {
    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(12) << "naive"
              << std::setw(12) << "tiled"
              << std::setw(16) << "naive_parallel"
              << std::setw(16) << "tiled_parallel"
              << "\n" << std::string(66, '-') << "\n";

    for (size_t sz : {32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u}) {
        double d0 = dfml::bench([](auto& a, auto& b){ return dfml::ops::matrix_multiply_naive(a, b, false); },       sz, sz, sz);
        double d1 = dfml::bench([](auto& a, auto& b){ return dfml::ops::matrix_multiply_avx2_tiled(a, b, false); },  sz, sz, sz);
        double d2 = dfml::bench([](auto& a, auto& b){ return dfml::ops::matrix_multiply_naive(a, b, true); },        sz, sz, sz);
        double d3 = dfml::bench([](auto& a, auto& b){ return dfml::ops::matrix_multiply_avx2_tiled(a, b, true); },   sz, sz, sz);

        std::cout << std::left  << std::setw(10) << sz
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(9)  << d0 << " ms"
                  << std::setw(9)  << d1 << " ms"
                  << std::setw(13) << d2 << " ms"
                  << std::setw(13) << d3 << " ms"
                  << "\n";
    }
}
