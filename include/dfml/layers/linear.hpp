#pragma once
 
#include "dfml/layers/layer.hpp"
#include "dfml/init/init.hpp"

namespace dfml::layers {

class Linear : public Layer {
public:
    Linear(size_t in_features, size_t out_features)
        : in_features_(in_features)
        , out_features_(out_features)
        , weights_({in_features, out_features}, true)
        , biases_({out_features}, true)
    {
        init_weights();
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        return ops::add_bias_to_matrix(ops::matrix_multiply(x, weights_), biases_);
    };

    std::vector<Tensor<float>> parameters() override {
        return {weights_, biases_};
    }

    Tensor<float>& weights() { return weights_; }
    Tensor<float>& biases() {return biases_; }

private:
    void init_weights() {
        init::xavier_normal(weights_, in_features_, out_features_);
        init::zeros(biases_);
    }

    size_t in_features_;
    size_t out_features_;
    Tensor<float> weights_;
    Tensor<float> biases_;
};

} //namespace dfml::layers