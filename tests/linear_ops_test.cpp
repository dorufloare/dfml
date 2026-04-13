#include "dfml/ops/add.hpp"
#include "dfml/ops/matrix_transpose.hpp"
#include "dfml/ops/matrix_multiply.hpp"

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

void check_close(float a, float b, const std::string& message, float eps = 1e-6f) {
    if (std::fabs(a - b) > eps) {
        throw std::runtime_error(message);
    }
}

void test_add_forward() {
    std::array<size_t, 2> shape{2, 3};
    dfml::Tensor<float> a(shape, std::vector<float>{1, 2, 3, 4, 5, 6});
    dfml::Tensor<float> b(shape, std::vector<float>{6, 5, 4, 3, 2, 1});
    dfml::Tensor<float> c = dfml::ops::add(a, b);
    check_close(c[0], 7.f, "add[0] mismatch");
    check_close(c[1], 7.f, "add[1] mismatch");
    check_close(c[2], 7.f, "add[2] mismatch");
    check_close(c[3], 7.f, "add[3] mismatch");
    check_close(c[4], 7.f, "add[4] mismatch");
    check_close(c[5], 7.f, "add[5] mismatch");
}

void test_add_bias_to_matrix() {
    std::array<size_t, 2> shape{2, 3};
    dfml::Tensor<float> a(shape, std::vector<float>{1, 2, 3, 4, 5, 6});
    dfml::Tensor<float> b({3}, std::vector<float>{10, 20, 30});
    dfml::Tensor<float> c = dfml::ops::add_bias_to_matrix(a, b);
    check_close(c[0], 11.f, "bias_add[0] mismatch");
    check_close(c[1], 22.f, "bias_add[1] mismatch");
    check_close(c[2], 33.f, "bias_add[2] mismatch");
    check_close(c[3], 14.f, "bias_add[3] mismatch");
    check_close(c[4], 25.f, "bias_add[4] mismatch");
    check_close(c[5], 36.f, "bias_add[5] mismatch");
}

void test_matrix_transpose() {
    std::array<size_t, 2> shape{2, 3};
    dfml::Tensor<float> a(shape, std::vector<float>{1, 2, 3, 4, 5, 6});
    dfml::Tensor<float> t = dfml::ops::matrix_transpose(a);
    check(t.nr_dimensions() == 2, "transpose rank mismatch");
    check(t.size(0) == 3 && t.size(1) == 2, "transpose shape mismatch");
    check_close(t[0], 1.f, "transpose[0] mismatch");
    check_close(t[1], 4.f, "transpose[1] mismatch");
    check_close(t[2], 2.f, "transpose[2] mismatch");
    check_close(t[3], 5.f, "transpose[3] mismatch");
    check_close(t[4], 3.f, "transpose[4] mismatch");
    check_close(t[5], 6.f, "transpose[5] mismatch");
}

void test_matmul_forward_values() {
    const std::array<size_t, 2> a_shape{2, 3};
    const std::array<size_t, 2> b_shape{3, 2};

    dfml::Tensor<float> a(a_shape, std::vector<float>{
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    });

    dfml::Tensor<float> b(b_shape, std::vector<float>{
        7.f, 8.f,
        9.f, 10.f,
        11.f, 12.f
    });

    dfml::Tensor<float> c = dfml::ops::matrix_multiply(a, b);

    check(c.nr_dimensions() == 2, "matmul output rank mismatch");
    check(c.size(0) == 2 && c.size(1) == 2, "matmul output shape mismatch");

    check_close(c[0], 58.f, "c[0] mismatch");
    check_close(c[1], 64.f, "c[1] mismatch");
    check_close(c[2], 139.f, "c[2] mismatch");
    check_close(c[3], 154.f, "c[3] mismatch");
}

void test_matmul_shape_checks() {
    bool threw = false;

    try {
        const std::array<size_t, 2> a_shape{2, 3};
        const std::array<size_t, 2> b_shape{4, 2};
        dfml::Tensor<float> a(a_shape);
        dfml::Tensor<float> b(b_shape);
        (void)dfml::ops::matrix_multiply(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "matmul should throw on mismatched inner dims");
}

void test_matmul_backward_local_gradients() {
    const std::array<size_t, 2> a_shape{2, 3};
    const std::array<size_t, 2> b_shape{3, 2};
    const std::array<size_t, 2> d_shape{2, 2};

    dfml::Tensor<float> a(a_shape, std::vector<float>{
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    }, true);

    dfml::Tensor<float> b(b_shape, std::vector<float>{
        7.f, 8.f,
        9.f, 10.f,
        11.f, 12.f
    }, true);

    dfml::Tensor<float> c = dfml::ops::matrix_multiply(a, b);

    dfml::Tensor<float> dc(d_shape, std::vector<float>{
        1.f, 1.f,
        1.f, 1.f
    });

    c.accumulate_grad(dc);
    c.backward();

    check(a.has_grad(), "a should have grad after backward");
    check(b.has_grad(), "b should have grad after backward");

    check_close(a.grad()[0], 15.f, "a.grad()[0] mismatch");
    check_close(a.grad()[1], 19.f, "a.grad()[1] mismatch");
    check_close(a.grad()[2], 23.f, "a.grad()[2] mismatch");
    check_close(a.grad()[3], 15.f, "a.grad()[3] mismatch");
    check_close(a.grad()[4], 19.f, "a.grad()[4] mismatch");
    check_close(a.grad()[5], 23.f, "a.grad()[5] mismatch");

    check_close(b.grad()[0], 5.f, "b.grad()[0] mismatch");
    check_close(b.grad()[1], 5.f, "b.grad()[1] mismatch");
    check_close(b.grad()[2], 7.f, "b.grad()[2] mismatch");
    check_close(b.grad()[3], 7.f, "b.grad()[3] mismatch");
    check_close(b.grad()[4], 9.f, "b.grad()[4] mismatch");
    check_close(b.grad()[5], 9.f, "b.grad()[5] mismatch");
}

}  // namespace

int main() {
    try {
        test_matmul_forward_values();
        test_matmul_shape_checks();
        test_matmul_backward_local_gradients();
        test_add_forward();
        test_add_bias_to_matrix();
        test_matrix_transpose();
        std::cout << "All linear ops tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Linear ops tests failed: " << e.what() << '\n';
        return 1;
    }
}
