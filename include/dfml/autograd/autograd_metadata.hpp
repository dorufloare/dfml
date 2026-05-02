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
    std::vector<Tensor<T>> grad_preallocated_workspaces;    //to avoid allocations in hot path (backward_fn)
};

}  // namespace dfml