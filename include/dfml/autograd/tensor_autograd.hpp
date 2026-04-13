#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "dfml/internal/tensor_impl.hpp"

namespace dfml {

template <typename T>
bool TensorImpl<T>::requires_grad() const noexcept {
	return requires_grad_;
}

template <typename T>
void TensorImpl<T>::set_requires_grad(bool v) {
	requires_grad_ = v;
	if (requires_grad_ && !autograd_metadata_) {
		autograd_metadata_ = std::make_unique<AutogradMetadata<T>>();
	}
	if (!requires_grad_) {
		autograd_metadata_.reset();
	}
}

template <typename T>
Tensor<T>& TensorImpl<T>::grad() {
	if (!autograd_metadata_)
		throw std::runtime_error("grad() called on tensor with requires_grad=false");
	if (!autograd_metadata_->grad.has_value())
		throw std::runtime_error("grad() requested before gradient was initialized");
	return autograd_metadata_->grad.value();
}

template <typename T>
const Tensor<T>& TensorImpl<T>::grad() const {
	if (!autograd_metadata_)
		throw std::runtime_error("grad() called on tensor with requires_grad=false");
	if (!autograd_metadata_->grad.has_value())
		throw std::runtime_error("grad() requested before gradient was initialized");
	return autograd_metadata_->grad.value();
}

template <typename T>
const std::vector<Tensor<T>>& TensorImpl<T>::previous_tensors() const {
	if (!autograd_metadata_)
		throw std::runtime_error("previous_tensors() called on tensor with requires_grad=false");
	return autograd_metadata_->previous_tensors;
}

template <typename T>
void TensorImpl<T>::set_previous_tensors(const std::vector<Tensor<T>>& tensors) {
	if (!autograd_metadata_)
		throw std::runtime_error("set_previous_tensors() called on tensor with requires_grad=false");
	autograd_metadata_->previous_tensors = tensors;
}

template <typename T>
void TensorImpl<T>::set_backward_function(const std::function<void()>& fn) {
	if (!autograd_metadata_)
		throw std::runtime_error("set_backward_function() called on tensor with requires_grad=false");
	autograd_metadata_->backward_function = fn;
}

template <typename T>
void TensorImpl<T>::backward() {
	if (!autograd_metadata_)
		throw std::runtime_error("backward() called on tensor with requires_grad=false");
	if (autograd_metadata_->backward_function) {
		autograd_metadata_->backward_function();
	}
}

template <typename T>
void TensorImpl<T>::zero_grad() {
	if (autograd_metadata_ && autograd_metadata_->grad.has_value()) {
		autograd_metadata_->grad->zero();
	}
}

template <typename T>
bool TensorImpl<T>::has_grad() const noexcept {
	return autograd_metadata_ && autograd_metadata_->grad.has_value();
}

template <typename T>
void TensorImpl<T>::accumulate_grad(const Tensor<T>& delta) {
	if (!requires_grad_) {
		return;
	}

	if (!autograd_metadata_)
		throw std::runtime_error("autograd_metadata_ should exist if requires_grad=true");

	if (delta.nr_elements() != nr_elements())
		throw std::invalid_argument("accumulate_grad: shape mismatch");

	if (!autograd_metadata_->grad.has_value()) {
		autograd_metadata_->grad = Tensor<T>(shape_, false);
		autograd_metadata_->grad->zero();
	}

	for (size_t i = 0; i < nr_elements(); ++i) {
		(*autograd_metadata_->grad)[i] += delta[i];
	}
}

template <typename T>
AutogradMetadata<T>* TensorImpl<T>::autograd_metadata() noexcept {
	return autograd_metadata_.get();
}

}  // namespace dfml
