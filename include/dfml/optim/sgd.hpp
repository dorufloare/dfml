#pragma once

#include <vector>
#include "dfml/optim/optimizer.hpp"

namespace dfml::optim {

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor<float>> parameters, float learning_rate=0.1f)
        : Optimizer(parameters)
        , learning_rate_(learning_rate)
    {

    }

    void step() override {
        for (auto& param : parameters_) {
            if (!param.has_grad()) continue;

            Tensor<float> grad = param.grad();
            float* grad_ptr = grad.data();
            float* param_ptr = param.data();

            for (size_t i = 0; i < param.nr_elements(); ++i) {
                param_ptr[i] -= learning_rate_ * grad_ptr[i];
            }
        }
    }

    void zero_grad() override {
        for (auto& param : parameters_) {
            param.zero_grad();
        }
    }
private:
    float learning_rate_;
};

} //namespace dfml::optim