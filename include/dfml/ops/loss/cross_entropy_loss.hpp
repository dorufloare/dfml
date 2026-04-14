#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>

#include "dfml/tensor.hpp"

namespace dfml::ops {


// probs = softmax(prediction)
// loss = -sum(log(probs)) / n

template<typename T>
Tensor<T> cross_entropy_loss(const Tensor<T>& logits, const std::vector<size_t>& labels) {
    //generalize to 2D
    if (logits.nr_dimensions() == 1)
        return cross_entropy_loss(logits.view({1, logits.size(0)}), labels);

    if (logits.nr_dimensions() != 2)
        throw std::invalid_argument("cross_entropy_loss: prediction must be 1D or 2D");

    const size_t M = logits.size(0);
    const size_t N = logits.size(1);

    if (labels.size() != M)
        throw std::invalid_argument("cross_entropy_loss: labels size must match batch size");

    const bool require_grad = GradGuard::is_grad_enabled() && logits.requires_grad();

    const T* logits_ptr = logits.data();

    std::vector<T> probs(N * M);
    for (size_t i = 0; i < M; ++i) {
        const T* logits_row = logits_ptr + i * N;
        T* probs_row = probs.data() + i * N;
        T row_max = *std::max_element(logits_row, logits_row + N);
        T sum{};

        for (size_t j = 0; j < N; ++j) {
            probs_row[j] = std::exp(logits_row[j] - row_max);
            sum += probs_row[j];
        }

        for (size_t j = 0; j < N; ++j) {
            probs_row[j] /= sum;
        }
    }

    T loss = T{};
    constexpr T eps = static_cast<T>(1e-12);  

    for (size_t i = 0; i < M; ++i) {
        loss -= std::log(probs[i * N + labels[i]] + eps);
    }
    loss /= static_cast<T>(M);

    Tensor<T> result = Tensor<T>::scalar(loss, require_grad);

    if (require_grad) {
        Tensor<T> logits_graph = logits;
        
        result.set_previous_tensors({logits_graph});

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([logits_graph, labels, probs, M, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T upstream = result_locked->grad()[0];

            Tensor<T> dlogits(logits_graph.shape());
            T* dlogits_ptr = dlogits.data();
            const T inv_M = upstream / static_cast<T>(M);
            const size_t N = logits_graph.size(1);

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    dlogits_ptr[i * N + j] = probs[i * N + j] * inv_M;
                    if (j == labels[i]) {
                        dlogits_ptr[i * N + j] -= inv_M;
                    }
                }
            }

            logits_graph.accumulate_grad(dlogits);
        });

    }

    return result;
}

template<typename T>
Tensor<T> cross_entropy_loss(const Tensor<T>& logits, std::initializer_list<size_t> labels) {
    return cross_entropy_loss(logits, std::vector<size_t>(labels));
}

template<typename T>
Tensor<T> cross_entropy_loss(const Tensor<T>& logits, const Tensor<T>& labels) {
    std::vector<size_t> idx(labels.nr_elements());
    for (size_t i = 0; i < labels.nr_elements(); ++i)
        idx[i] = static_cast<size_t>(labels[i]);
    return cross_entropy_loss(logits, idx);
}

} //namespace dfml::ops
