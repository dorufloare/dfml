#pragma once

#include "dfml/tensor.hpp"

namespace dfml::ops {

//relu(x) = max(0, x)

template<typename T>
Tensor<T> relu(const Tensor<T>& a) {
    const bool require_grad = GradGuard::is_grad_enabled() && a.requires_grad();
    Tensor<T> result(a.shape(), require_grad);

    const T* a_ptr = a.data();
    T* result_ptr = result.data();

    const size_t N = a.nr_elements();
    for (size_t i = 0; i < N; ++i) {
        result_ptr[i] = a_ptr[i] > T{} ? a_ptr[i] : T{};
    }

    if (require_grad) {
        Tensor<T> a_graph = a;

        result.set_previous_tensors({a_graph});

        const auto result_weak = result.make_weak_tensor();

        result.set_backward_function([a_graph, N, result_weak]() mutable {
            auto result_locked = Tensor<T>::lock_weak_tensor(result_weak);
            if (!result_locked.has_value()) return;

            const T* dr_ptr = result_locked->grad().data();
            const T* a_ptr = a_graph.data();

            Tensor<T> dA(a_graph.shape());
            T* da_ptr = dA.data();

            //positive -> pass grad, negative -> zero
            for (size_t i = 0; i < N; ++i) {
                da_ptr[i] = a_ptr[i] > T{} ? dr_ptr[i] : T{}; 
            }

            a_graph.accumulate_grad(dA);
        });

    }

    return result;
}

} //namespace dfml::ops