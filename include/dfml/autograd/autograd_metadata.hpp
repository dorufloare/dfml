#pragma once

#include <functional>
#include <optional>
#include <vector>

namespace dfml {

template<typename T> class Tensor;

template<typename T> 
struct AutogradMetadata {
    std::optional<Tensor<T>> grad;
    std::function<void()> backward_function = nullptr;
    std::vector<Tensor<T>> previous_tensors;
};

}  // namespace dfml