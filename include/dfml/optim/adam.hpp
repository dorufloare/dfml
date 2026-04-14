#pragma once

#include <vector>
#include <cmath>

#include "dfml/optim/optimizer.hpp"

namespace dfml::optim {

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor<float>> parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : Optimizer(std::move(parameters))
        , learning_rate_(lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , epsilon_(epsilon)
        , beta1_pow(1)
        , beta2_pow(1)
        , timestep_(0)
    {
        for (auto& param : parameters_) {
            Tensor<float> m(param.shape());
            m.zero();
            first_moment_.push_back(m);

            Tensor<float> v(param.shape());
            v.zero();
            second_moment_.push_back(v);
        }
    }

    void step() override {
        timestep_++;
        beta1_pow *= beta1_;
        beta2_pow *= beta2_;
        float bc1 = 1.f - beta1_pow;
        float bc2 = 1.f - beta2_pow;

       for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param.has_grad()) continue;

            Tensor<float> grad = param.grad();
            float* grad_ptr = grad.data();
            float* param_ptr = param.data();
            float* fm_ptr = first_moment_[i].data();
            float *sm_ptr = second_moment_[i].data();

            for (size_t j = 0; j < param.nr_elements(); ++j) {
                float g = grad_ptr[j];

                fm_ptr[j] = beta1_ * fm_ptr[j] + (1.f - beta1_) * g;
                sm_ptr[j] = beta2_ * sm_ptr[j] + (1.f - beta2_) * g * g;

                float fm_hat = fm_ptr[j] / bc1;
                float sm_hat = sm_ptr[j] / bc2;

                param_ptr[j] -= learning_rate_ * fm_hat / (std::sqrt(sm_hat) + epsilon_);
            }
        }
    }

    void zero_grad() override {
        for (auto& param : parameters_) {
            param.zero_grad();
        }
    }
private:
    std::vector<Tensor<float>> first_moment_;
    std::vector<Tensor<float>> second_moment_;

    float learning_rate_;
    float beta1_, beta2_, epsilon_;
    float beta1_pow, beta2_pow;
    int timestep_;

};

} //namespace dfml::optim