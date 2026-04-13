#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>

#include "dfml/tensor.hpp"

namespace dfml::ops {

template<typename T>
Tensor<T> mse_loss(const Tensor<T>& prediction, const Tensor<T>& target) {
    if (prediction.shape() != target.shape())
        throw std::invalid_argument("mse_loss: prediction and target shapes must match");

    const bool require_grad = GradGuard::is_grad_enabled() && prediction.requires_grad();

    const size_t N = prediction.nr_elements();
    const T* prediction_ptr = prediction.data();
    const T* target_ptr = target.data();

    T loss = T{};
    for (size_t i = 0; i < N; ++i) {
        T diff = prediction_ptr[i] - target_ptr[i];
        loss += diff * diff;
    }
    loss /= static_cast<T>(N);

    Tensor<T> result = Tensor<T>::scalar(loss, require_grad);

    if (require_grad) {
        Tensor<T> prediction_graph = prediction;
        
        result.set_previous_tensors({prediction_graph});

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([prediction_graph, target, N, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T upstream = result_locked->grad()[0];

            Tensor<T> dpred(prediction_graph.shape());
            T* dpred_ptr = dpred.data();
            const T* prediction_ptr = prediction_graph.data();
            const T* target_ptr = target.data();
            const T scale = upstream * T{2} / static_cast<T>(N);

            for (size_t i = 0; i < N; ++i) {
                dpred_ptr[i] = scale * (prediction_ptr[i] - target_ptr[i]);
            }

            prediction_graph.accumulate_grad(dpred);
        });

    }

    return result;
}

} //namespace dfml::ops
