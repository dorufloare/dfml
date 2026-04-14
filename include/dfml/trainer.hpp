#pragma once

#include "dfml/tensor.hpp"
#include "dfml/layers/layers.hpp"
#include "dfml/optim/optim.hpp"
#include "dfml/autograd/grad_guard.hpp"
#include <iostream>

namespace dfml {
    
using MetricFn = std::function<float(const Tensor<float>&, const Tensor<float>&)>;

class Trainer {
public:
    Trainer(layers::Sequential& model, optim::Optimizer& optimizer, ops::LossFn loss_fn)
        : model_(model)
        , optimizer_(optimizer)
        , loss_fn_(loss_fn)
    {

    }

    Tensor<float> fit(const Tensor<float>& X, const Tensor<float>& Y, size_t epochs, size_t print_every = 100) {
        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            auto prediction = model_.forward(X);
            auto loss = loss_fn_(prediction, Y);
            loss.backward();
            optimizer_.step();
            optimizer_.zero_grad();
            if (epoch % print_every == 0) {
                std::cout << "epoch " << epoch << "  loss: " << loss.data()[0];
                for (auto& [name, fn] : metrics_)
                    std::cout << "  " << name << ": " << fn(prediction, Y);
                std::cout << "\n";
            }
        }
        dfml::GradGuard guard;  
        return model_.forward(X);
    }

    Tensor<float> predict(const Tensor<float>& X) {
        GradGuard guard;
        return model_.forward(X);
    }

    void add_metric(const std::string& name, MetricFn metric) {
        metrics_.push_back({name, metric});
    }

private:
    layers::Sequential& model_;
    optim::Optimizer& optimizer_;
    ops::LossFn loss_fn_;

    std::vector<std::pair<std::string, MetricFn>> metrics_;
};

} //namespace dfml