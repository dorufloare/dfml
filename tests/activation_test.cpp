#include "dfml/ops/activation/relu.hpp"
#include "dfml/ops/activation/sigmoid.hpp"
#include "dfml/ops/activation/tanh.hpp"
#include "dfml/ops/activation/softmax.hpp"

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

void check_close(float a, float b, const std::string& message, float eps = 1e-5f) {
    if (std::fabs(a - b) > eps) {
        throw std::runtime_error(
            message + " (got " + std::to_string(a) + ", expected " + std::to_string(b) + ")");
    }
}

// ── ReLU ─────────────────────────────────────────────────────────────────────

void test_relu_forward_positive() {
    dfml::Tensor<float> a({3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> r = dfml::ops::relu(a);
    check_close(r[0], 1.f, "relu: positive input unchanged [0]");
    check_close(r[1], 2.f, "relu: positive input unchanged [1]");
    check_close(r[2], 3.f, "relu: positive input unchanged [2]");
}

void test_relu_forward_negative() {
    dfml::Tensor<float> a({3}, std::vector<float>{-1.f, -2.f, -3.f});
    dfml::Tensor<float> r = dfml::ops::relu(a);
    check_close(r[0], 0.f, "relu: negative clamped to 0 [0]");
    check_close(r[1], 0.f, "relu: negative clamped to 0 [1]");
    check_close(r[2], 0.f, "relu: negative clamped to 0 [2]");
}

void test_relu_forward_mixed() {
    dfml::Tensor<float> a({5}, std::vector<float>{-2.f, -1.f, 0.f, 1.f, 2.f});
    dfml::Tensor<float> r = dfml::ops::relu(a);
    check_close(r[0], 0.f, "relu mixed [0]");
    check_close(r[1], 0.f, "relu mixed [1]");
    check_close(r[2], 0.f, "relu mixed [2]");
    check_close(r[3], 1.f, "relu mixed [3]");
    check_close(r[4], 2.f, "relu mixed [4]");
}

void test_relu_does_not_modify_input() {
    dfml::Tensor<float> a({3}, std::vector<float>{-1.f, 0.f, 1.f});
    (void)dfml::ops::relu(a);
    check_close(a[0], -1.f, "relu must not modify input [0]");
    check_close(a[1],  0.f, "relu must not modify input [1]");
    check_close(a[2],  1.f, "relu must not modify input [2]");
}

void test_relu_backward() {
    // a = [-2, -1, 0, 1, 2], upstream grad = all-ones
    // expected grad_a: [0, 0, 0, 1, 1]
    dfml::Tensor<float> a({5}, std::vector<float>{-2.f, -1.f, 0.f, 1.f, 2.f}, true);
    dfml::Tensor<float> r = dfml::ops::relu(a);

    dfml::Tensor<float> dr({5}, std::vector<float>{1.f, 1.f, 1.f, 1.f, 1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(a.has_grad(), "relu backward: a must have grad");
    check_close(a.grad()[0], 0.f, "relu backward grad[0]");
    check_close(a.grad()[1], 0.f, "relu backward grad[1]");
    check_close(a.grad()[2], 0.f, "relu backward grad[2]");
    check_close(a.grad()[3], 1.f, "relu backward grad[3]");
    check_close(a.grad()[4], 1.f, "relu backward grad[4]");
}

void test_relu_no_grad_when_not_required() {
    dfml::Tensor<float> a({3}, std::vector<float>{-1.f, 0.f, 1.f}, false);
    dfml::Tensor<float> r = dfml::ops::relu(a);
    check(!r.requires_grad(), "relu: result should not require grad when input does not");
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────

void test_sigmoid_forward_known_values() {
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.73106, sigmoid(-1) ≈ 0.26894
    dfml::Tensor<float> a({3}, std::vector<float>{0.f, 1.f, -1.f});
    dfml::Tensor<float> r = dfml::ops::sigmoid(a);
    check_close(r[0], 0.5f,      "sigmoid(0) != 0.5");
    check_close(r[1], 0.73106f,  "sigmoid(1) mismatch", 1e-4f);
    check_close(r[2], 0.26894f,  "sigmoid(-1) mismatch", 1e-4f);
}

void test_sigmoid_output_range() {
    dfml::Tensor<float> a({4}, std::vector<float>{-10.f, -1.f, 1.f, 10.f});
    dfml::Tensor<float> r = dfml::ops::sigmoid(a);
    for (size_t i = 0; i < 4; ++i) {
        check(r[i] > 0.f && r[i] < 1.f, "sigmoid output must be in (0,1)");
    }
}

void test_sigmoid_symmetry() {
    // sigmoid(-x) + sigmoid(x) == 1
    dfml::Tensor<float> pos({1}, std::vector<float>{2.f});
    dfml::Tensor<float> neg({1}, std::vector<float>{-2.f});
    dfml::Tensor<float> rp = dfml::ops::sigmoid(pos);
    dfml::Tensor<float> rn = dfml::ops::sigmoid(neg);
    check_close(rp[0] + rn[0], 1.f, "sigmoid symmetry: sigmoid(x)+sigmoid(-x) != 1");
}

void test_sigmoid_does_not_modify_input() {
    dfml::Tensor<float> a({2}, std::vector<float>{1.f, -1.f});
    (void)dfml::ops::sigmoid(a);
    check_close(a[0],  1.f, "sigmoid must not modify input [0]");
    check_close(a[1], -1.f, "sigmoid must not modify input [1]");
}

void test_sigmoid_backward() {
    // sigmoid(0)=0.5, dsigmoid(0)/dx = 0.5*0.5 = 0.25
    // sigmoid(1)≈0.73106, dsigmoid/dx ≈ 0.73106*(1-0.73106) ≈ 0.19661
    dfml::Tensor<float> a({2}, std::vector<float>{0.f, 1.f}, true);
    dfml::Tensor<float> r = dfml::ops::sigmoid(a);

    dfml::Tensor<float> dr({2}, std::vector<float>{1.f, 1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(a.has_grad(), "sigmoid backward: a must have grad");
    check_close(a.grad()[0], 0.25f,   "sigmoid backward grad[0]", 1e-5f);
    check_close(a.grad()[1], 0.19661f,"sigmoid backward grad[1]", 1e-4f);
}

void test_sigmoid_backward_scaled_upstream() {
    // upstream grad = 2 → output grad should be 2 * local_grad
    dfml::Tensor<float> a({1}, std::vector<float>{0.f}, true);
    dfml::Tensor<float> r = dfml::ops::sigmoid(a);

    dfml::Tensor<float> dr({1}, std::vector<float>{2.f});
    r.accumulate_grad(dr);
    r.backward();

    check_close(a.grad()[0], 0.5f, "sigmoid backward: scaled upstream grad", 1e-5f);
}

// ── Tanh ─────────────────────────────────────────────────────────────────────

void test_tanh_forward_known_values() {
    // tanh(0)=0, tanh(1)≈0.76159, tanh(-1)≈-0.76159
    dfml::Tensor<float> a({3}, std::vector<float>{0.f, 1.f, -1.f});
    dfml::Tensor<float> r = dfml::ops::tanh(a);
    check_close(r[0],  0.f,      "tanh(0) != 0");
    check_close(r[1],  0.76159f, "tanh(1) mismatch", 1e-4f);
    check_close(r[2], -0.76159f, "tanh(-1) mismatch", 1e-4f);
}

void test_tanh_output_range() {
    // Use moderate inputs: float tanh saturates to exactly ±1.f for large |x|
    dfml::Tensor<float> a({4}, std::vector<float>{-3.f, -1.f, 1.f, 3.f});
    dfml::Tensor<float> r = dfml::ops::tanh(a);
    for (size_t i = 0; i < 4; ++i) {
        check(r[i] > -1.f && r[i] < 1.f, "tanh output must be in (-1,1)");
    }
}

void test_tanh_antisymmetry() {
    // tanh(-x) == -tanh(x)
    dfml::Tensor<float> pos({1}, std::vector<float>{1.5f});
    dfml::Tensor<float> neg({1}, std::vector<float>{-1.5f});
    dfml::Tensor<float> rp = dfml::ops::tanh(pos);
    dfml::Tensor<float> rn = dfml::ops::tanh(neg);
    check_close(rp[0] + rn[0], 0.f, "tanh antisymmetry: tanh(x)+tanh(-x) != 0");
}

void test_tanh_does_not_modify_input() {
    dfml::Tensor<float> a({2}, std::vector<float>{1.f, -1.f});
    (void)dfml::ops::tanh(a);
    check_close(a[0],  1.f, "tanh must not modify input [0]");
    check_close(a[1], -1.f, "tanh must not modify input [1]");
}

void test_tanh_backward() {
    // dtanh(0)/dx = 1 - tanh(0)^2 = 1
    // dtanh(1)/dx = 1 - 0.76159^2 ≈ 1 - 0.58002 = 0.41998
    dfml::Tensor<float> a({2}, std::vector<float>{0.f, 1.f}, true);
    dfml::Tensor<float> r = dfml::ops::tanh(a);

    dfml::Tensor<float> dr({2}, std::vector<float>{1.f, 1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(a.has_grad(), "tanh backward: a must have grad");
    check_close(a.grad()[0], 1.f,     "tanh backward grad[0]", 1e-5f);
    check_close(a.grad()[1], 0.41998f,"tanh backward grad[1]", 1e-4f);
}

void test_tanh_backward_scaled_upstream() {
    // upstream = 3, x = 0 → grad = 3 * 1 = 3
    dfml::Tensor<float> a({1}, std::vector<float>{0.f}, true);
    dfml::Tensor<float> r = dfml::ops::tanh(a);

    dfml::Tensor<float> dr({1}, std::vector<float>{3.f});
    r.accumulate_grad(dr);
    r.backward();

    check_close(a.grad()[0], 3.f, "tanh backward: scaled upstream grad", 1e-5f);
}

// ── Softmax ──────────────────────────────────────────────────────────────────

void test_softmax_rejects_non_2d() {
    bool threw = false;
    try {
        dfml::Tensor<float> a({3}, std::vector<float>{1.f, 2.f, 3.f});
        (void)dfml::ops::softmax(a);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "softmax should throw for 1D input");
}

void test_softmax_forward_sums_to_one() {
    // Each row of a softmax output must sum to 1
    dfml::Tensor<float> a({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::softmax(a);

    float row0_sum = r[0] + r[1] + r[2];
    float row1_sum = r[3] + r[4] + r[5];
    check_close(row0_sum, 1.f, "softmax row 0 must sum to 1", 1e-5f);
    check_close(row1_sum, 1.f, "softmax row 1 must sum to 1", 1e-5f);
}

void test_softmax_forward_all_positive() {
    dfml::Tensor<float> a({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::softmax(a);
    for (size_t i = 0; i < r.nr_elements(); ++i) {
        check(r[i] > 0.f, "softmax output must be strictly positive");
    }
}

void test_softmax_forward_uniform_input() {
    // Equal inputs → equal outputs (1/N per element)
    dfml::Tensor<float> a({1, 4}, std::vector<float>{0.f, 0.f, 0.f, 0.f});
    dfml::Tensor<float> r = dfml::ops::softmax(a);
    for (size_t i = 0; i < 4; ++i) {
        check_close(r[i], 0.25f, "softmax uniform: each element should be 0.25", 1e-5f);
    }
}

void test_softmax_forward_known_values() {
    // Row 0: [1,2,3] → [0.09003, 0.24473, 0.66524]
    // Row 1: [1,1,1] → [0.33333, 0.33333, 0.33333]
    dfml::Tensor<float> a({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::softmax(a);
    check_close(r[0], 0.09003f, "softmax[0][0]", 1e-4f);
    check_close(r[1], 0.24473f, "softmax[0][1]", 1e-4f);
    check_close(r[2], 0.66524f, "softmax[0][2]", 1e-4f);
    check_close(r[3], 0.33333f, "softmax[1][0]", 1e-4f);
    check_close(r[4], 0.33333f, "softmax[1][1]", 1e-4f);
    check_close(r[5], 0.33333f, "softmax[1][2]", 1e-4f);
}

void test_softmax_invariant_to_shift() {
    // softmax(x + c) == softmax(x)
    dfml::Tensor<float> a({1, 3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> b({1, 3}, std::vector<float>{101.f, 102.f, 103.f});
    dfml::Tensor<float> ra = dfml::ops::softmax(a);
    dfml::Tensor<float> rb = dfml::ops::softmax(b);
    check_close(ra[0], rb[0], "softmax shift invariance [0]", 1e-5f);
    check_close(ra[1], rb[1], "softmax shift invariance [1]", 1e-5f);
    check_close(ra[2], rb[2], "softmax shift invariance [2]", 1e-5f);
}

void test_softmax_does_not_modify_input() {
    dfml::Tensor<float> a({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    (void)dfml::ops::softmax(a);
    check_close(a[0], 1.f, "softmax must not modify input [0]");
    check_close(a[1], 2.f, "softmax must not modify input [1]");
    check_close(a[2], 3.f, "softmax must not modify input [2]");
    check_close(a[3], 4.f, "softmax must not modify input [3]");
}

void test_softmax_backward_allones_upstream_gives_zero_grad() {
    // When upstream gradient is uniform (all ones), the softmax Jacobian
    // contracts to zero: dL/dX[i][j] = s[i][j] * (1 - sum(s[i])) = 0
    dfml::Tensor<float> a({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f}, true);
    dfml::Tensor<float> r = dfml::ops::softmax(a);

    dfml::Tensor<float> dr({2, 3}, std::vector<float>{1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(a.has_grad(), "softmax backward: a must have grad");
    for (size_t i = 0; i < 6; ++i) {
        check_close(a.grad()[i], 0.f,
            "softmax backward: uniform upstream must yield zero grad at [" + std::to_string(i) + "]",
            1e-5f);
    }
}

void test_softmax_backward_known_values() {
    // Input: [[2,1],[0,0]], upstream: [[1,0],[1,0]]
    // softmax row0 ≈ [0.73106, 0.26894]
    // softmax row1 = [0.5, 0.5]
    // grad row0: dot=0.73106
    //   da[0][0] = 0.73106*(1-0.73106) ≈ 0.19661
    //   da[0][1] = 0.26894*(0-0.73106) ≈ -0.19661
    // grad row1: dot=0.5
    //   da[1][0] = 0.5*(1-0.5) = 0.25
    //   da[1][1] = 0.5*(0-0.5) = -0.25
    dfml::Tensor<float> a({2, 2}, std::vector<float>{2.f, 1.f, 0.f, 0.f}, true);
    dfml::Tensor<float> r = dfml::ops::softmax(a);

    dfml::Tensor<float> dr({2, 2}, std::vector<float>{1.f, 0.f, 1.f, 0.f});
    r.accumulate_grad(dr);
    r.backward();

    check(a.has_grad(), "softmax backward known: a must have grad");
    check_close(a.grad()[0],  0.19661f, "softmax backward da[0][0]", 1e-4f);
    check_close(a.grad()[1], -0.19661f, "softmax backward da[0][1]", 1e-4f);
    check_close(a.grad()[2],  0.25f,    "softmax backward da[1][0]", 1e-5f);
    check_close(a.grad()[3], -0.25f,    "softmax backward da[1][1]", 1e-5f);
}

}  // namespace

int main() {
    try {
        // ReLU
        test_relu_forward_positive();
        test_relu_forward_negative();
        test_relu_forward_mixed();
        test_relu_does_not_modify_input();
        test_relu_backward();
        test_relu_no_grad_when_not_required();

        // Sigmoid
        test_sigmoid_forward_known_values();
        test_sigmoid_output_range();
        test_sigmoid_symmetry();
        test_sigmoid_does_not_modify_input();
        test_sigmoid_backward();
        test_sigmoid_backward_scaled_upstream();

        // Tanh
        test_tanh_forward_known_values();
        test_tanh_output_range();
        test_tanh_antisymmetry();
        test_tanh_does_not_modify_input();
        test_tanh_backward();
        test_tanh_backward_scaled_upstream();

        // Softmax
        test_softmax_rejects_non_2d();
        test_softmax_forward_sums_to_one();
        test_softmax_forward_all_positive();
        test_softmax_forward_uniform_input();
        test_softmax_forward_known_values();
        test_softmax_invariant_to_shift();
        test_softmax_does_not_modify_input();
        test_softmax_backward_allones_upstream_gives_zero_grad();
        test_softmax_backward_known_values();

        std::cout << "All activation tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Activation tests failed: " << e.what() << '\n';
        return 1;
    }
}
