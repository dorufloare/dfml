#pragma once
 
#include "dfml/layers/layer.hpp"

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
        //He initialization: std = sqrt(2 / in)

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.f, std::sqrt(2.f / static_cast<float>(in_features_)));

        float* weights_ptr = weights_.data();
        for (size_t i = 0; i < weights_.nr_elements(); ++i) {
            weights_ptr[i] = dist(rng);
        }

        biases_.zero();
    }

    size_t in_features_;
    size_t out_features_;
    Tensor<float> weights_;
    Tensor<float> biases_;
};

} //namespace dfml::layers