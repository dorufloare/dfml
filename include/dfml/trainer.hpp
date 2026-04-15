#pragma once

#include "dfml/tensor.hpp"
#include "dfml/layers/layers.hpp"
#include "dfml/optim/optim.hpp"
#include "dfml/autograd/grad_guard.hpp"
#include "dfml/data/data_loader.hpp"
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

    void print_metrics(size_t epoch, const Tensor<float>& pred, const Tensor<float>& Y, float loss_val) {
        std::cout << "epoch " << epoch << "  loss: " << loss_val;
        for (auto& [name, fn] : metrics_)
            std::cout << "  " << name << ": " << fn(pred, Y);
        std::cout << "\n";
    }

    void print_metrics(size_t epoch, const Tensor<float>& X, const Tensor<float>& Y) {
        GradGuard guard;
        auto pred = model_.forward(X);
        auto loss = loss_fn_(pred, Y);
        print_metrics(epoch, pred, Y, loss.data()[0]);
    }

    Tensor<float> fit(const Tensor<float>& X, const Tensor<float>& Y, size_t epochs, size_t batch_size = 0, size_t print_every = 100) {
        if (batch_size == 0) {
            for (size_t epoch = 1; epoch <= epochs; ++epoch) {
                optimizer_.zero_grad();
                auto prediction = model_.forward(X);
                auto loss = loss_fn_(prediction, Y);
                loss.backward();
                optimizer_.step();

                if (epoch % print_every == 0) {
                    print_metrics(epoch, prediction, Y, loss.data()[0]);
                }
            }
        } else {
            for (size_t epoch = 1; epoch <= epochs; ++epoch) {
                DataLoader<float> loader(X, Y, batch_size, true);
                for (const auto& [X_batch, Y_batch] : loader) {
                    optimizer_.zero_grad();
                    auto prediction = model_.forward(X_batch);
                    auto loss = loss_fn_(prediction, Y_batch);
                    loss.backward();
                    optimizer_.step();
                }

                if (epoch % print_every == 0) {
                    print_metrics(epoch, X, Y);
                }
            }
        }
        
        GradGuard guard;  
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