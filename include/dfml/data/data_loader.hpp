#pragma once

#include "dfml/tensor.hpp"
#include "dfml/utils/random.hpp"

namespace dfml {

template<typename T>
class DataLoader {
public:
    DataLoader(const Tensor<T>& X, const Tensor<T>& Y, size_t batch_size, bool shuffle = true)
        : X_(X)
        , Y_(Y)
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , n_(X.size(0))
    {
        if (X.size(0) != Y.size(0))
            throw std::invalid_argument("DataLoader: X and Y must have same number of rows");
        if (batch_size == 0)
            throw std::invalid_argument("DataLoader: batch_size must be > 0");
 
        indices_.resize(X.size(0));
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    class Iterator {
    public:
         Iterator(const Tensor<T>& X, const Tensor<T>& Y,
                 const std::vector<size_t>& indices,
                 size_t batch_size,
                 size_t pos)
            : X_(X)
            , Y_(Y)
            , indices_(indices)
            , batch_size_(batch_size)
            , pos_(pos)
        {}

        bool operator != (const Iterator& oth) const {
            return pos_ < oth.pos_;
        }

        Iterator& operator++ () {
            pos_ += batch_size_;
            return *this;
        }

        std::pair<Tensor<T>,Tensor<T>> operator*() const {
            size_t n = indices_.size();
            size_t end = std::min(pos_ + batch_size_, n);
            size_t count = end - pos_;
            
            std::vector<size_t> batch_indices(
                indices_.begin() + pos_,
                indices_.begin() + pos_ + count
            );
 
            return {
                X_.index_select(batch_indices),
                Y_.index_select(batch_indices)
            };
        }

    private:
        const Tensor<T>& X_;
        const Tensor<T>& Y_;
        const std::vector<size_t>& indices_;
        size_t batch_size_;
        size_t pos_;
    };

    Iterator begin() {
        std::shuffle(indices_.begin(), indices_.end(), global_rng());
        return Iterator(X_, Y_, indices_, batch_size_, 0);
    }

    Iterator end() const noexcept {
        return Iterator(X_, Y_, indices_, batch_size_, n_);
    }

    size_t num_batches() const {
        return (n_ + batch_size_ - 1) / batch_size_;
    }
 
    size_t size() const { return n_; }
private:
    Tensor<T> X_;
    Tensor<T> Y_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t n_;
};

}