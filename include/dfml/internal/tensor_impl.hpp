#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dfml/internal/storage.hpp"
#include "dfml/autograd/autograd_metadata.hpp"

namespace dfml {

template <typename T>
class Tensor;

template<typename T>
class TensorImpl  {
public:
    TensorImpl(std::span<const size_t> shape, bool requires_grad = false) 
        : storage_(std::make_shared<Storage<T>>(compute_size_from_shape(shape)))
        , shape_(shape.begin(), shape.end())
        , requires_grad_(requires_grad) {
        if (requires_grad_) {
            autograd_metadata_ = std::make_unique<AutogradMetadata<T>>();
        }
    }

    TensorImpl(std::span<const size_t> shape, std::vector<T> data, bool requires_grad = false) 
        : storage_(std::make_shared<Storage<T>>(std::move(data)))
        , shape_(shape.begin(), shape.end())
        , requires_grad_(requires_grad) {
        assert(storage_->size() == compute_size_from_shape(shape) && "data/shape size mismatch");
        if (requires_grad_) {
            autograd_metadata_ = std::make_unique<AutogradMetadata<T>>();
        }
    }

    static Tensor<T> scalar(T value, bool requires_grad = false) {
        const std::array<size_t, 1> shape{1};
        Tensor<T> t(shape, requires_grad);
        t[0] = value;
        return t;
    }

    // SHAPE
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    void set_shape(const std::vector<size_t>& shape) { shape_ = shape; }

    size_t nr_dimensions() const noexcept { return shape_.size(); }

    size_t size(size_t dim) const {
        if (dim >= nr_dimensions()) {
            throw std::out_of_range("dim out of range");
        }
        return shape_[dim];
    }

    // DATA
    T* data() noexcept { return storage_->data(); }
    const T* data() const noexcept { return storage_->data(); }
    const Storage<T>& get_storage() const noexcept { return *storage_; }
    void set_storage(const Storage<T>& storage) { *storage_ = storage; }

    size_t nr_elements() const noexcept { return storage_->size(); }

    TensorImpl<T> index_select(const std::vector<size_t>& indices) const {
        size_t n = indices.size();
        size_t row_stride = nr_elements() / size(0);

        std::vector<size_t> new_shape = shape();
        new_shape[0] = n;

        TensorImpl<T> out(new_shape);

        for (size_t i = 0; i < n; ++i) {
            std::copy(
                data() + indices[i] * row_stride,
                data() + indices[i] * row_stride + row_stride,
                out.data() + i * row_stride
            );
        }
        return out;
    }

    // DATA ACCESS
    T operator[](size_t i) const { return (*storage_)[i]; }
    T& operator[](size_t i) { return (*storage_)[i]; }

    T& at(std::span<const size_t> position) {
        return (*storage_)[get_flat_index(position)];
    }

    const T& at(std::span<const size_t> position) const {
        return (*storage_)[get_flat_index(position)];
    }

    // UTILS
    std::shared_ptr<TensorImpl<T>> view(std::span<const size_t> new_shape) {
        return std::shared_ptr<TensorImpl<T>>(
            new TensorImpl<T>(storage_, std::vector<size_t>(new_shape.begin(), new_shape.end()))
        );
    }

    std::shared_ptr<TensorImpl<T>> flatten() const {
        return view(std::array<size_t, 1>{nr_elements()});
    }

    std::shared_ptr<TensorImpl<T>> clone() const {
        auto new_impl = std::make_shared<TensorImpl<T>>(shape_);
        new_impl->set_storage(*storage_);
        return new_impl;
    }

    void fill(T value) {
        std::fill(storage_->data(), storage_->data() + nr_elements(), value);
    }

    void zero() { fill(T{}); }

    // autograd
    bool requires_grad() const noexcept;
    void set_requires_grad(bool v);

    // grad accessors
    Tensor<T>& grad();
    const Tensor<T>& grad() const;
    bool has_grad() const noexcept;
    void zero_grad();
    void accumulate_grad(const Tensor<T>& delta);

    // previous_tensors accessors
    const std::vector<Tensor<T>>& previous_tensors() const;
    void set_previous_tensors(const std::vector<Tensor<T>>& tensors);

    // backward function
    void set_backward_function(const std::function<void()>& fn);
    void backward();

    AutogradMetadata<T>* autograd_metadata() noexcept;

private:
    friend class Tensor<T>;

    //for view
    TensorImpl(std::shared_ptr<Storage<T>>& storage, std::vector<size_t> shape)
        : storage_(storage)
        , shape_(std::move(shape))
        , requires_grad_(false)
    {}

    size_t get_flat_index(std::span<const size_t> position) const {
        assert(position.size() == nr_dimensions() && "index rank mistmatch");
        size_t flat_index = 0, stride = 1;
        for (int i = (int)nr_dimensions() - 1; i >= 0; --i) {
            flat_index += position[i] * stride;
            stride *= shape_[i];
        }
        return flat_index;
    }

    size_t compute_size_from_shape(std::span<const size_t> shape) const {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
    }

    std::shared_ptr<Storage<T>> storage_;
    std::vector<size_t> shape_;

    bool requires_grad_;
    std::unique_ptr<AutogradMetadata<T>> autograd_metadata_ = nullptr;
};

}  // namespace dfml