#pragma once

#include "dfml/tensor.hpp"

namespace dfml::ops {

template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add: input shapes must match");
    }
    const bool require_grad = GradGuard::is_grad_enabled() || (a.requires_grad() || b.requires_grad());
    Tensor<T> result(a.shape(), require_grad);

    const T* a_ptr = a.data();
    const T* b_ptr = b.data();
    T* result_ptr = result.data();

    for (size_t i = 0; i < a.nr_elements(); ++i) {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    if (require_grad) {
        Tensor<T> a_graph = a;
        Tensor<T> b_graph = b;

        result.set_previous_tensors({a_graph, b_graph});

        auto result_weak = result.make_weak_tensor();

        result.set_backward_function([a_graph, b_graph, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            // dR / dA = d(A + B) / dA = 1 => dL / dA = 1 * dL / dR

            if (a_graph.requires_grad()) a_graph.accumulate_grad(result_locked->grad());
            if (b_graph.requires_grad()) b_graph.accumulate_grad(result_locked->grad());
        });
    }

    return result;
}


//adds b in each row of a
template<typename T>
Tensor<T> add_bias_to_matrix(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.nr_dimensions() != 2) 
        throw std::invalid_argument("add_bias_to_matrix: first input must be a matrix (2 dimensions)");
    
    if (b.nr_dimensions() != 1) 
        throw std::invalid_argument("add_bias_to_matrix: first input must be a vector (1 dimension)");
    
    if (b.nr_elements() != a.size(1)) 
        throw std::invalid_argument("add_bias_to_matrix: input shapes must match");
    
    const size_t M = a.size(0);
    const size_t N = a.size(1);

    const bool require_grad = GradGuard::is_grad_enabled() || (a.requires_grad() || b.requires_grad());

    Tensor<T> result(a.shape(), require_grad);

    const T* a_ptr = a.data();
    const T* b_ptr = b.data();
    T* result_ptr = result.data();

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            result_ptr[i * N + j] = a_ptr[i * N + j] + b_ptr[j];

    if (require_grad) {
        auto result_weak = result.make_weak_tensor();

        Tensor<T> a_graph = a;
        Tensor<T> b_graph = b;

        result.set_previous_tensors({a_graph, b_graph});

        result.set_backward_function([a_graph, b_graph, M, N, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;
            
            if (a_graph.requires_grad()) a_graph.accumulate_grad(result_locked->grad());

            if (b_graph.requires_grad()) {
                // dL / dB[j] = sum(dL / dR[i][j], i=0..M)
                Tensor<T> dB({N});
                dB.zero();

                T* db_ptr = dB.data();
                const T* dr_ptr = result_locked->grad().data(); 

                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < N; ++j)
                        db_ptr[j] += dr_ptr[i * N + j];
                
                b_graph.accumulate_grad(dB);
            }
        });
    }

    return result;
}


} //namespace dfml::ops