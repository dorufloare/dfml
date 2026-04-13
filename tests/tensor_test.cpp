#include "dfml/tensor.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_construction_and_shape() {
    const std::array<size_t, 2> shape{2, 3};
    dfml::Tensor<float> t(shape);
    check(t.nr_dimensions() == 2, "nr_dimensions mismatch");
    check(t.size(0) == 2, "size(0) mismatch");
    check(t.size(1) == 3, "size(1) mismatch");
    check(t.nr_elements() == 6, "nr_elements mismatch");

    for (size_t i = 0; i < t.nr_elements(); ++i) {
        check(t[i] == 0.0f, "default value is not zero");
    }

    bool threw = false;
    try {
        (void)t.size(2);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "size(dim) should throw for invalid dim");
}

void test_data_constructor_and_at() {
    const std::array<size_t, 2> shape{2, 3};
    dfml::Tensor<float> t(shape, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    check(t.at(std::array<size_t, 2>{0, 0}) == 1.f, "at(0,0) mismatch");
    check(t.at(std::array<size_t, 2>{0, 2}) == 3.f, "at(0,2) mismatch");
    check(t.at(std::array<size_t, 2>{1, 0}) == 4.f, "at(1,0) mismatch");
    check(t.at(std::array<size_t, 2>{1, 2}) == 6.f, "at(1,2) mismatch");
}

void test_view_shares_memory() {
    const std::array<size_t, 2> original_shape{2, 3};
    const std::array<size_t, 2> reshaped_shape{3, 2};
    dfml::Tensor<float> original(original_shape, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    dfml::Tensor<float> reshaped = original.view(reshaped_shape);

    check(reshaped.nr_dimensions() == 2, "view rank mismatch");
    check(reshaped.size(0) == 3 && reshaped.size(1) == 2, "view shape mismatch");

    reshaped[5] = 99.f;
    check(original[5] == 99.f, "view should share storage with original");
}

void test_clone_is_deep_copy() {
    const std::array<size_t, 2> shape{2, 2};
    dfml::Tensor<float> original(shape, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    dfml::Tensor<float> copied = original.clone();

    copied[0] = 42.f;
    check(original[0] == 1.f, "clone should not modify original storage");
    check(copied[0] == 42.f, "clone write failed");
}

void test_fill_and_zero() {
    const std::array<size_t, 1> shape{5};
    dfml::Tensor<float> t(shape);
    t.fill(7.f);
    for (size_t i = 0; i < t.nr_elements(); ++i) {
        check(t[i] == 7.f, "fill() mismatch");
    }

    t.zero();
    for (size_t i = 0; i < t.nr_elements(); ++i) {
        check(t[i] == 0.f, "zero() mismatch");
    }
}

void test_scalar() {
    dfml::Tensor<float> s = dfml::Tensor<float>::scalar(3.14f);
    check(s.nr_elements() == 1, "scalar nr_elements mismatch");
    check(std::fabs(s[0] - 3.14f) < 1e-6f, "scalar value mismatch");
}

void test_autograd_basics() {
    const std::array<size_t, 1> shape{3};
    dfml::Tensor<float> a(shape, true);
    check(a.requires_grad(), "requires_grad should be true");
    check(!a.has_grad(), "has_grad should be false initially");

    dfml::Tensor<float> delta(shape, std::vector<float>{1.f, 2.f, 3.f});
    a.accumulate_grad(delta);
    check(a.has_grad(), "has_grad should be true after accumulate_grad");
    check(a.grad()[0] == 1.f && a.grad()[1] == 2.f && a.grad()[2] == 3.f,
          "grad after first accumulate mismatch");

    a.accumulate_grad(delta);
    check(a.grad()[0] == 2.f && a.grad()[1] == 4.f && a.grad()[2] == 6.f,
          "grad after second accumulate mismatch");

    a.zero_grad();
    check(a.grad()[0] == 0.f && a.grad()[1] == 0.f && a.grad()[2] == 0.f,
          "zero_grad mismatch");

    bool called = false;
    a.set_backward_function([&called]() { called = true; });
    a.backward();
    check(called, "backward_function callback not called");

    a.set_previous_tensors({delta});
    check(a.previous_tensors().size() == 1, "previous_tensors size mismatch");

    dfml::Tensor<float> b(shape, false);
    b.accumulate_grad(delta);
    check(!b.has_grad(), "non-grad tensor should ignore accumulate_grad");
}

}  // namespace

int main() {
    try {
        test_construction_and_shape();
        test_data_constructor_and_at();
        test_view_shares_memory();
        test_clone_is_deep_copy();
        test_fill_and_zero();
        test_scalar();
        test_autograd_basics();
        std::cout << "All tensor tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Tensor tests failed: " << e.what() << '\n';
        return 1;
    }
}