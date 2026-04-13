#include "dfml/optim/sgd.hpp"
#include "dfml/layers/linear.hpp"
#include "dfml/layers/sequential.hpp"
#include "dfml/ops/loss/mse_loss.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_close(float a, float b, const std::string& message, float eps = 1e-5f) {
    if (std::fabs(a - b) > eps)
        throw std::runtime_error(
            message + " (got " + std::to_string(a) + ", expected " + std::to_string(b) + ")");
}

// ── SGD step ─────────────────────────────────────────────────────────────────

void test_sgd_step_updates_param() {
    // param = [2.0], grad = [1.0], lr = 0.1
    // after step: param = 2.0 - 0.1 * 1.0 = 1.9
    dfml::Tensor<float> param({1}, std::vector<float>{2.f}, true);
    dfml::Tensor<float> g({1}, std::vector<float>{1.f});
    param.accumulate_grad(g);

    dfml::optim::SGD sgd({param}, 0.1f);
    sgd.step();

    check_close(param[0], 1.9f, "sgd step: param not updated correctly");
}

void test_sgd_step_multiple_params() {
    // Two params, different grads
    dfml::Tensor<float> p0({2}, std::vector<float>{1.f, 2.f}, true);
    dfml::Tensor<float> p1({2}, std::vector<float>{3.f, 4.f}, true);

    dfml::Tensor<float> g0({2}, std::vector<float>{0.5f, 1.f});
    dfml::Tensor<float> g1({2}, std::vector<float>{2.f, 0.f});
    p0.accumulate_grad(g0);
    p1.accumulate_grad(g1);

    dfml::optim::SGD sgd({p0, p1}, 0.1f);
    sgd.step();

    check_close(p0[0], 1.f - 0.1f * 0.5f, "sgd multi: p0[0]");
    check_close(p0[1], 2.f - 0.1f * 1.f,  "sgd multi: p0[1]");
    check_close(p1[0], 3.f - 0.1f * 2.f,  "sgd multi: p1[0]");
    check_close(p1[1], 4.f - 0.1f * 0.f,  "sgd multi: p1[1]");
}

void test_sgd_skips_param_without_grad() {
    // Param with no grad must not be touched
    dfml::Tensor<float> param({2}, std::vector<float>{5.f, 6.f}, true);

    dfml::optim::SGD sgd({param}, 0.5f);
    sgd.step();  // no grad set — must be a no-op

    check_close(param[0], 5.f, "sgd: param without grad must not change [0]");
    check_close(param[1], 6.f, "sgd: param without grad must not change [1]");
}

void test_sgd_zero_grad_clears_grad() {
    dfml::Tensor<float> param({2}, std::vector<float>{1.f, 1.f}, true);
    dfml::Tensor<float> g({2}, std::vector<float>{3.f, 4.f});
    param.accumulate_grad(g);
    check(param.has_grad(), "sgd zero_grad: param should have grad before zero_grad");

    dfml::optim::SGD sgd({param}, 0.1f);
    sgd.zero_grad();

    check_close(param.grad()[0], 0.f, "sgd zero_grad: grad[0] must be zero");
    check_close(param.grad()[1], 0.f, "sgd zero_grad: grad[1] must be zero");
}

void test_sgd_step_twice_accumulates() {
    // Two steps with the same grad: param -= 2 * lr * grad
    dfml::Tensor<float> param({1}, std::vector<float>{1.f}, true);
    dfml::optim::SGD sgd({param}, 0.1f);

    for (int i = 0; i < 2; ++i) {
        dfml::Tensor<float> g({1}, std::vector<float>{1.f});
        param.accumulate_grad(g);
        sgd.step();
        sgd.zero_grad();
    }

    check_close(param[0], 0.8f, "sgd two steps: param should be 0.8", 1e-5f);
}

void test_sgd_learning_rate_zero_no_update() {
    dfml::Tensor<float> param({2}, std::vector<float>{3.f, 4.f}, true);
    dfml::Tensor<float> g({2}, std::vector<float>{10.f, 10.f});
    param.accumulate_grad(g);

    dfml::optim::SGD sgd({param}, 0.f);
    sgd.step();

    check_close(param[0], 3.f, "sgd lr=0: param must not change [0]");
    check_close(param[1], 4.f, "sgd lr=0: param must not change [1]");
}

// ── SGD + Linear end-to-end ───────────────────────────────────────────────────

void test_sgd_linear_loss_decreases() {
    // Single Linear(1,1): learn to map [1] → [2]
    // With enough steps, MSE loss should strictly decrease
    dfml::layers::Linear linear(1, 1);
    linear.weights().data()[0] = 0.f;
    linear.biases().data()[0]  = 0.f;

    dfml::optim::SGD sgd(linear.parameters(), 0.1f);

    dfml::Tensor<float> x({1, 1}, std::vector<float>{1.f});
    dfml::Tensor<float> y({1, 1}, std::vector<float>{2.f});

    float prev_loss = 1e9f;
    for (int step = 0; step < 20; ++step) {
        auto out  = linear.forward(x);
        auto loss = dfml::ops::mse_loss(out, y);
        loss.backward();
        sgd.step();
        sgd.zero_grad();

        check(loss[0] < prev_loss, "sgd linear: loss must decrease at step " + std::to_string(step));
        prev_loss = loss[0];
    }
}

void test_sgd_linear_converges() {
    // Same setup: after enough steps the output should be close to 2
    dfml::layers::Linear linear(1, 1);
    linear.weights().data()[0] = 0.f;
    linear.biases().data()[0]  = 0.f;

    dfml::optim::SGD sgd(linear.parameters(), 0.1f);

    dfml::Tensor<float> x({1, 1}, std::vector<float>{1.f});
    dfml::Tensor<float> y({1, 1}, std::vector<float>{2.f});

    for (int step = 0; step < 100; ++step) {
        auto out  = linear.forward(x);
        auto loss = dfml::ops::mse_loss(out, y);
        loss.backward();
        sgd.step();
        sgd.zero_grad();
    }

    auto out = linear.forward(x);
    check_close(out[0], 2.f, "sgd linear: output should converge to 2", 0.01f);
}

}  // namespace

int main() {
    try {
        test_sgd_step_updates_param();
        test_sgd_step_multiple_params();
        test_sgd_skips_param_without_grad();
        test_sgd_zero_grad_clears_grad();
        test_sgd_step_twice_accumulates();
        test_sgd_learning_rate_zero_no_update();
        test_sgd_linear_loss_decreases();
        test_sgd_linear_converges();

        std::cout << "All optimizer tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Optimizer tests failed: " << e.what() << '\n';
        return 1;
    }
}
