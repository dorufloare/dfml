#pragma once
 
#include "dfml/layers/layer.hpp"
#include <memory>

namespace dfml::layers {

class Sequential : public Layer {
public:
    template<typename LayerType, typename... Args>
    Sequential& add(Args&&... args) {
        layers_.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
        return *this;
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        Tensor<float> result = x;
        for (auto& layer : layers_)
            result = layer->forward(result);
        return result;
    };

    std::vector<Tensor<float>> parameters() override {
        std::vector<Tensor<float>> params;
        for (auto& layer : layers_) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} //namespace dfml::layers