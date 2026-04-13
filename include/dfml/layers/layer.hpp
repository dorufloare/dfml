#pragma once
 
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <random>

#include "dfml/tensor.hpp"
#include "dfml/ops/ops.hpp"

namespace dfml::layers {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor<float> forward(const Tensor<float>& x) = 0;
    virtual std::vector<Tensor<float>> parameters() = 0;

    void zero_grad() {
        for (auto& p : parameters())
            p.zero_grad();
    }
private:

};

} //namespace dfml::layers