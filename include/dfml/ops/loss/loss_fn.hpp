#pragma once

#include <functional>
#include "dfml/tensor.hpp"

namespace dfml::ops {

using LossFn = std::function<Tensor<float>(const Tensor<float>&, const Tensor<float>&)>;

} // namespace dfml::ops
