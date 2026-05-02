#pragma once

#include "dfml/tensor.hpp"
#include <cmath>

namespace dfml::ops {

// fast tanh aproximation
inline float fast_tanh(float x) {
    x = std::max(-4.f, std::min(4.f, x));
    float x2 = x * x;
    return x * (27.f + x2) / (27.f + 9.f * x2);  
}

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
        result_ptr[i] = fast_tanh(a_ptr[i]);
    }

    if (require_grad) {
        Tensor<T> a_graph = a;

        result.set_previous_tensors({a_graph});

        result.add_grad_preallocated_workspace(Tensor<T>(a_graph.shape()));

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([a_graph, N, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T* dr_ptr = result_locked->grad().data();
            const T* r_ptr = result_locked->data();

            Tensor<T> dA = result_locked->get_grad_preallocated_workspace(0);
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