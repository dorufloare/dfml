#pragma once

#include "dfml/tensor.hpp"
#include <cmath>

namespace dfml::ops {

//r = tanh(x)
//dL/dx = dL/dR * dR/dX = dL/dR * (1 - r^2)

template<typename T>
Tensor<T> tanh(const Tensor<T>& a) {
    const bool require_grad = GradGuard::is_grad_enabled() && a.requires_grad();
    Tensor<T> result(a.shape(), require_grad);

    const T* a_ptr = a.data();
    T* result_ptr = result.data();

    const size_t N = a.nr_elements();
    for (size_t i = 0; i < N; ++i) {
        result_ptr[i] = std::tanh(a_ptr[i]);
    }

    if (require_grad) {
        Tensor<T> a_graph = a;

        result.set_previous_tensors({a_graph});

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([a_graph, N, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T* dr_ptr = result_locked->grad().data();
            const T* r_ptr = result_locked->data();

            Tensor<T> dA(a_graph.shape());
            T* da_ptr = dA.data();

            // dr/dx = 1 - r*r
            for (size_t i = 0; i < N; ++i) {
                da_ptr[i] = dr_ptr[i] * (T{1} - r_ptr[i] * r_ptr[i]); 
            }

            a_graph.accumulate_grad(dA);
        });

    }

    return result;
}

} //namespace dfml::ops