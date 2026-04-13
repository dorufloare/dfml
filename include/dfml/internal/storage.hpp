#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace dfml {

template<typename T>
struct Storage {
    std::vector<T> buffer_;

    explicit Storage(size_t size) : buffer_(size, T{}) {}
    explicit Storage(std::vector<T> data) : buffer_(std::move(data)) {}

    Storage(const Storage&) = default;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(const Storage&) = default;
    Storage& operator=(Storage&&) noexcept = default;

    T& operator[](size_t i) { return buffer_[i]; }
    const T& operator[](size_t i) const { return buffer_[i]; }

    size_t size() const noexcept { return buffer_.size(); }

    T* data() noexcept { return buffer_.data(); }
    const T* data() const noexcept { return buffer_.data(); }
};

}  // namespace dfml