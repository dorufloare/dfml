#include "dfml/ops/loss/mse_loss.hpp"
#include "dfml/ops/loss/cross_entropy_loss.hpp"

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

// ── MSE Loss ─────────────────────────────────────────────────────────────────

void test_mse_zero_loss() {
    // identical pred and target → loss = 0
    dfml::Tensor<float> pred({3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> tgt ({3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);
    check_close(r[0], 0.f, "mse_loss: identical inputs must give 0");
}

void test_mse_forward_known_values() {
    // pred=[0,0], target=[1,1] → loss = (1+1)/2 = 1.0
    dfml::Tensor<float> pred({2}, std::vector<float>{0.f, 0.f});
    dfml::Tensor<float> tgt ({2}, std::vector<float>{1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);
    check_close(r[0], 1.f, "mse_loss: known value [0,0] vs [1,1]");
}

void test_mse_forward_asymmetric() {
    // pred=[1,3], target=[1,1] → loss = (0 + 4) / 2 = 2.0
    dfml::Tensor<float> pred({2}, std::vector<float>{1.f, 3.f});
    dfml::Tensor<float> tgt ({2}, std::vector<float>{1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);
    check_close(r[0], 2.f, "mse_loss: asymmetric known value");
}

void test_mse_shape_mismatch_throws() {
    bool threw = false;
    try {
        dfml::Tensor<float> pred({2}, std::vector<float>{1.f, 2.f});
        dfml::Tensor<float> tgt ({3}, std::vector<float>{1.f, 2.f, 3.f});
        (void)dfml::ops::mse_loss(pred, tgt);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "mse_loss: shape mismatch must throw");
}

void test_mse_no_grad_when_not_required() {
    dfml::Tensor<float> pred({2}, std::vector<float>{1.f, 2.f}, false);
    dfml::Tensor<float> tgt ({2}, std::vector<float>{1.f, 2.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);
    check(!r.requires_grad(), "mse_loss: result should not require grad when input does not");
}

void test_mse_backward() {
    // pred=[1,3], target=[1,1]
    // dpred[0] = 2*(1-1)/2 = 0
    // dpred[1] = 2*(3-1)/2 = 2
    dfml::Tensor<float> pred({2}, std::vector<float>{1.f, 3.f}, true);
    dfml::Tensor<float> tgt ({2}, std::vector<float>{1.f, 1.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);

    dfml::Tensor<float> dr({1}, std::vector<float>{1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(pred.has_grad(), "mse_loss backward: pred must have grad");
    check_close(pred.grad()[0], 0.f, "mse_loss backward: dpred[0]");
    check_close(pred.grad()[1], 2.f, "mse_loss backward: dpred[1]");
}

void test_mse_backward_scaled_upstream() {
    // pred=[0], target=[1], upstream=3
    // dpred = 3 * 2*(0-1)/1 = -6
    dfml::Tensor<float> pred({1}, std::vector<float>{0.f}, true);
    dfml::Tensor<float> tgt ({1}, std::vector<float>{1.f});
    dfml::Tensor<float> r = dfml::ops::mse_loss(pred, tgt);

    dfml::Tensor<float> dr({1}, std::vector<float>{3.f});
    r.accumulate_grad(dr);
    r.backward();

    check_close(pred.grad()[0], -6.f, "mse_loss backward: scaled upstream", 1e-5f);
}

// ── Cross Entropy Loss ────────────────────────────────────────────────────────

void test_ce_forward_single_class_certain() {
    // logits=[0,0,10], label=2: prob[2] ≈ 1.0 → loss ≈ 0
    dfml::Tensor<float> logits({1, 3}, std::vector<float>{0.f, 0.f, 10.f});
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {2});
    check(r[0] < 0.001f, "ce_loss: near-certain prediction should have near-zero loss");
}

void test_ce_forward_uniform_logits() {
    // Equal logits over N classes → loss = log(N)
    // N=4: log(4) ≈ 1.38629
    dfml::Tensor<float> logits({1, 4}, std::vector<float>{0.f, 0.f, 0.f, 0.f});
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {0});
    check_close(r[0], std::log(4.f), "ce_loss: uniform logits → loss = log(N)", 1e-4f);
}

void test_ce_forward_known_values() {
    // logits=[[1,2,3]], label=[2]
    // probs ≈ [0.09003, 0.24473, 0.66524]
    // loss = -log(0.66524) ≈ 0.40760
    dfml::Tensor<float> logits({1, 3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {2});
    check_close(r[0], 0.40760f, "ce_loss: known value [1,2,3] label=2", 1e-4f);
}

void test_ce_forward_batch() {
    // logits=[[1,2],[3,4]], labels=[0,1]
    // Both rows have same structure (shifted by 2), same probs [0.26894, 0.73106]
    // loss_0 = -log(0.26894) ≈ 1.31326, loss_1 = -log(0.73106) ≈ 0.31326
    // mean ≈ 0.81326
    dfml::Tensor<float> logits({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {0, 1});
    check_close(r[0], 0.81326f, "ce_loss: batch known value", 1e-4f);
}

void test_ce_forward_1d_input_generalizes() {
    // 1D input [1,2,3] with label [2] should give same result as 2D [[1,2,3]]
    dfml::Tensor<float> logits_1d({3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> logits_2d({1, 3}, std::vector<float>{1.f, 2.f, 3.f});
    dfml::Tensor<float> r1 = dfml::ops::cross_entropy_loss(logits_1d, {2});
    dfml::Tensor<float> r2 = dfml::ops::cross_entropy_loss(logits_2d, {2});
    check_close(r1[0], r2[0], "ce_loss: 1D and 2D inputs must give same result", 1e-5f);
}

void test_ce_invalid_dim_throws() {
    bool threw = false;
    try {
        // 3D tensor should throw
        dfml::Tensor<float> logits({2, 2, 2}, std::vector<float>(8, 1.f));
        (void)dfml::ops::cross_entropy_loss(logits, {0, 0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "ce_loss: 3D input must throw");
}

void test_ce_labels_size_mismatch_throws() {
    bool threw = false;
    try {
        dfml::Tensor<float> logits({2, 3}, std::vector<float>(6, 1.f));
        (void)dfml::ops::cross_entropy_loss(logits, {0});  // need 2 labels, gave 1
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "ce_loss: label count mismatch must throw");
}

void test_ce_no_grad_when_not_required() {
    dfml::Tensor<float> logits({1, 3}, std::vector<float>{1.f, 2.f, 3.f}, false);
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {2});
    check(!r.requires_grad(), "ce_loss: result should not require grad when input does not");
}

void test_ce_backward_single_row() {
    // logits=[[1,2,3]], label=[2]
    // probs ≈ [0.09003, 0.24473, 0.66524], M=1
    // dlogits[0] = (0.09003 - 0) / 1 = 0.09003
    // dlogits[1] = (0.24473 - 0) / 1 = 0.24473
    // dlogits[2] = (0.66524 - 1) / 1 = -0.33476
    dfml::Tensor<float> logits({1, 3}, std::vector<float>{1.f, 2.f, 3.f}, true);
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {2});

    dfml::Tensor<float> dr({1}, std::vector<float>{1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(logits.has_grad(), "ce_loss backward: logits must have grad");
    check_close(logits.grad()[0],  0.09003f, "ce_loss backward: dlogits[0]", 1e-4f);
    check_close(logits.grad()[1],  0.24473f, "ce_loss backward: dlogits[1]", 1e-4f);
    check_close(logits.grad()[2], -0.33476f, "ce_loss backward: dlogits[2]", 1e-4f);
}

void test_ce_backward_batch() {
    // logits=[[1,2],[3,4]], labels=[0,1], M=2
    // Row 0 probs ≈ [0.26894, 0.73106], correct=0
    //   dlogits[0][0] = (0.26894 - 1) / 2 = -0.36553
    //   dlogits[0][1] = (0.73106 - 0) / 2 =  0.36553
    // Row 1 probs ≈ [0.26894, 0.73106], correct=1
    //   dlogits[1][0] = (0.26894 - 0) / 2 =  0.13447
    //   dlogits[1][1] = (0.73106 - 1) / 2 = -0.13447
    dfml::Tensor<float> logits({2, 2}, std::vector<float>{1.f, 2.f, 3.f, 4.f}, true);
    dfml::Tensor<float> r = dfml::ops::cross_entropy_loss(logits, {0, 1});

    dfml::Tensor<float> dr({1}, std::vector<float>{1.f});
    r.accumulate_grad(dr);
    r.backward();

    check(logits.has_grad(), "ce_loss backward batch: logits must have grad");
    check_close(logits.grad()[0], -0.36553f, "ce_loss backward batch: dlogits[0][0]", 1e-4f);
    check_close(logits.grad()[1],  0.36553f, "ce_loss backward batch: dlogits[0][1]", 1e-4f);
    check_close(logits.grad()[2],  0.13447f, "ce_loss backward batch: dlogits[1][0]", 1e-4f);
    check_close(logits.grad()[3], -0.13447f, "ce_loss backward batch: dlogits[1][1]", 1e-4f);
}

}  // namespace

int main() {
    try {
        // MSE Loss
        test_mse_zero_loss();
        test_mse_forward_known_values();
        test_mse_forward_asymmetric();
        test_mse_shape_mismatch_throws();
        test_mse_no_grad_when_not_required();
        test_mse_backward();
        test_mse_backward_scaled_upstream();

        // Cross Entropy Loss
        test_ce_forward_single_class_certain();
        test_ce_forward_uniform_logits();
        test_ce_forward_known_values();
        test_ce_forward_batch();
        test_ce_forward_1d_input_generalizes();
        test_ce_invalid_dim_throws();
        test_ce_labels_size_mismatch_throws();
        test_ce_no_grad_when_not_required();
        test_ce_backward_single_row();
        test_ce_backward_batch();

        std::cout << "All loss tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Loss tests failed: " << e.what() << '\n';
        return 1;
    }
}
