#pragma once

#include <vector>
#include "dfml/tensor.hpp"

namespace dfml::optim {

class Optimizer {
public:
    Optimizer(std::vector<Tensor<float>> parameters)
        : parameters_(std::move(parameters))
    {

    }
    
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
protected:
    std::vector<Tensor<float>> parameters_;
};

} //namespace dfml::optim