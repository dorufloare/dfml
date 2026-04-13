#pragma once
 
#include "dfml/layers/layer.hpp"

namespace dfml::layers {

class ReLU : public Layer {
public:
    Tensor<float> forward(const Tensor<float>& x) override {
        return ops::relu(x);
    }
 
    std::vector<Tensor<float>> parameters() override {
        return {}; 
    }
};

class Sigmoid : public Layer {
public:
    Tensor<float> forward(const Tensor<float>& x) override {
        return ops::sigmoid(x);
    }
 
    std::vector<Tensor<float>> parameters() override {
        return {};
    }
};

class Tanh : public Layer {
public:
    Tensor<float> forward(const Tensor<float>& x) override {
        return ops::tanh(x);
    }

    std::vector<Tensor<float>> parameters() override {
        return {};
    }
};

class Softmax : public Layer {
public:
    Tensor<float> forward(const Tensor<float>& x) override {
        return ops::softmax(x);
    }

    std::vector<Tensor<float>> parameters() override {
        return {};
    }
};

} //namespace dfml::layers