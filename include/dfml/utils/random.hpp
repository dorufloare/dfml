#pragma once
#include <random>

namespace dfml {

inline std::mt19937& global_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

inline void set_rng_seed(uint32_t seed) {
    global_rng().seed(seed);
}

}