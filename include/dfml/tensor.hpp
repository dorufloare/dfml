#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dfml/internal/tensor_impl.hpp"
#include "dfml/autograd/grad_guard.hpp"

namespace dfml {

template <typename T>
class Tensor {
public:
    Tensor(std::span<const size_t> shape, bool requires_grad = false)
        : impl_(std::make_shared<TensorImpl<T>>(shape, requires_grad)) {}

    Tensor(std::initializer_list<size_t> shape, bool requires_grad = false)
        : impl_(std::make_shared<TensorImpl<T>>(
              std::span<const size_t>(shape.begin(), shape.size()), requires_grad)) {}

    Tensor(std::span<const size_t> shape, std::vector<T> data, bool requires_grad = false)
        : impl_(std::make_shared<TensorImpl<T>>(shape, std::move(data), requires_grad)) {}

    Tensor(std::initializer_list<size_t> shape, std::vector<T> data, bool requires_grad = false)
        : impl_(std::make_shared<TensorImpl<T>>(
              std::span<const size_t>(shape.begin(), shape.size()), std::move(data), requires_grad)) {}

    static Tensor scalar(T value, bool requires_grad = false) {
        return TensorImpl<T>::scalar(value, requires_grad);
    }

    const std::vector<size_t>& shape() const noexcept { return impl_->shape(); }
    void set_shape(const std::vector<size_t>& shape) { impl_->set_shape(shape); }

    TensorImpl<T>* get_raw_impl_ptr() const { return impl_.get(); }

    size_t nr_dimensions() const noexcept { return impl_->nr_dimensions(); }
    size_t size(size_t dim) const { return impl_->size(dim); }
    size_t nr_elements() const noexcept { return impl_->nr_elements(); }

    T* data() noexcept { return impl_->data(); }
    const T* data() const noexcept { return impl_->data(); }
    const Storage<T>& get_storage() const noexcept { return impl_->get_storage(); }
    void set_storage(const Storage<T>& storage) { impl_->set_storage(storage); }

    T operator[](size_t i) const { return (*impl_)[i]; }
    T& operator[](size_t i) { return (*impl_)[i]; }

    T& at(std::span<const size_t> position) { return impl_->at(position); }
    const T& at(std::span<const size_t> position) const { return impl_->at(position); }

    Tensor view(std::span<const size_t> new_shape) const { return Tensor(impl_->view(new_shape)); }
    Tensor view(std::initializer_list<size_t> new_shape) const {
        return Tensor(impl_->view(std::span<const size_t>(new_shape.begin(), new_shape.size())));
    }
    Tensor clone() const { return Tensor(impl_->clone()); }

    void fill(T value) { impl_->fill(value); }
    void zero() { impl_->zero(); }

    bool requires_grad() const noexcept { return impl_->requires_grad(); }
    void set_requires_grad(bool v) { impl_->set_requires_grad(v); }

    // grad accessors
    const Tensor& grad() const { return impl_->grad(); }
    Tensor& grad() { return impl_->grad(); }
    bool has_grad() const noexcept { return impl_->has_grad(); }
    void zero_grad() { impl_->zero_grad(); }
    void accumulate_grad(const Tensor& delta) { impl_->accumulate_grad(delta); }

    // previous_tensors accessors
    const std::vector<Tensor>& previous_tensors() const { return impl_->previous_tensors(); }
    void set_previous_tensors(const std::vector<Tensor>& tensors) { impl_->set_previous_tensors(tensors); }

    // backward function
    void set_backward_function(const std::function<void()>& fn) { impl_->set_backward_function(fn); }

    void backward() {
        if (!has_grad())
            accumulate_grad(Tensor<T>::scalar(T{1}));

        std::vector<Tensor<T>> topo;
        std::unordered_set<TensorImpl<T>*> visited;

        std::function<void(const Tensor<T>&)> build_topo = [&](const Tensor<T>& t) {
            if (!t.requires_grad()) return;
            if (visited.count(t.impl_.get())) return;
            visited.insert(t.impl_.get());
            for (auto& prev : t.previous_tensors())
                build_topo(prev);
            topo.push_back(t);
        };

        build_topo(*this);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            it->impl_->backward();
    }

    std::weak_ptr<TensorImpl<T>> make_weak_tensor() const noexcept { return impl_; }

    static std::optional<Tensor> lock_weak_tensor(const std::weak_ptr<TensorImpl<T>>& weak) {
        auto shared = weak.lock();
        if (!shared) {
            return std::nullopt;
        }
        return Tensor(std::move(shared));
    }

private:
    friend class TensorImpl<T>;

    Tensor(std::shared_ptr<TensorImpl<T>> impl) 
        : impl_(std::move(impl)) {}

    std::shared_ptr<TensorImpl<T>> impl_;
};

}  // namespace dfml

#include "dfml/autograd/tensor_autograd.hpp"