#include "dfml/layers/linear.hpp"
#include "dfml/layers/activation.hpp"
#include "dfml/layers/sequential.hpp"

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

// ── Linear ────────────────────────────────────────────────────────────────────

void test_linear_output_shape() {
    dfml::layers::Linear linear(3, 4);
    dfml::Tensor<float> x({2, 3}, std::vector<float>(6, 1.f));
    auto out = linear.forward(x);
    check(out.nr_dimensions() == 2,       "linear output must be 2D");
    check(out.size(0) == 2,               "linear output rows must match batch size");
    check(out.size(1) == 4,               "linear output cols must match out_features");
}

void test_linear_parameters_count() {
    dfml::layers::Linear linear(3, 4);
    check(linear.parameters().size() == 2, "linear must have exactly 2 parameters (weights + biases)");
}

void test_linear_biases_initialized_to_zero() {
    dfml::layers::Linear linear(4, 3);
    const auto& b = linear.biases();
    for (size_t i = 0; i < b.nr_elements(); ++i)
        check_close(b[i], 0.f, "linear bias[" + std::to_string(i) + "] must be zero at init");
}

void test_linear_forward_known_values() {
    // weights {3,2}: [[1,0],[0,1],[0,0]], biases [0,0]
    // x = [[1,2,3]] → output = [[1,2]]
    dfml::layers::Linear linear(3, 2);
    float* w = linear.weights().data();
    w[0]=1.f; w[1]=0.f;
    w[2]=0.f; w[3]=1.f;
    w[4]=0.f; w[5]=0.f;
    linear.biases().zero();

    dfml::Tensor<float> x({1, 3}, std::vector<float>{1.f, 2.f, 3.f});
    auto out = linear.forward(x);

    check_close(out[0], 1.f, "linear forward known: out[0]");
    check_close(out[1], 2.f, "linear forward known: out[1]");
}

void test_linear_forward_with_bias() {
    // weights {2,2}: [[1,0],[0,1]], biases [10,20]
    // x = [[3,4]] → output = [[13, 24]]
    dfml::layers::Linear linear(2, 2);
    float* w = linear.weights().data();
    w[0]=1.f; w[1]=0.f;
    w[2]=0.f; w[3]=1.f;
    float* b = linear.biases().data();
    b[0]=10.f; b[1]=20.f;

    dfml::Tensor<float> x({1, 2}, std::vector<float>{3.f, 4.f});
    auto out = linear.forward(x);

    check_close(out[0], 13.f, "linear with bias: out[0]");
    check_close(out[1], 24.f, "linear with bias: out[1]");
}

void test_linear_backward() {
    // weights {2,1}: [[1],[1]], biases [0]
    // x = [[1,2]] → output = [[3]]
    // upstream = 1
    // dx = dout @ W^T = [[1]] @ [[1,1]] = [[1,1]]
    // dW = x^T @ dout = [[1],[2]] @ [[1]] = [[1],[2]]
    // db = [1]
    dfml::layers::Linear linear(2, 1);
    float* w = linear.weights().data();
    w[0]=1.f; w[1]=1.f;
    linear.biases().zero();

    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 2.f}, true);
    auto out = linear.forward(x);

    dfml::Tensor<float> dr({1, 1}, std::vector<float>{1.f});
    out.accumulate_grad(dr);
    out.backward();

    check(x.has_grad(),                   "linear backward: x must have grad");
    check_close(x.grad()[0], 1.f,         "linear backward: dx[0]");
    check_close(x.grad()[1], 1.f,         "linear backward: dx[1]");
    check(linear.weights().has_grad(),    "linear backward: weights must have grad");
    check_close(linear.weights().grad()[0], 1.f, "linear backward: dW[0]");
    check_close(linear.weights().grad()[1], 2.f, "linear backward: dW[1]");
    check(linear.biases().has_grad(),     "linear backward: biases must have grad");
    check_close(linear.biases().grad()[0], 1.f,  "linear backward: db[0]");
}

void test_linear_zero_grad() {
    dfml::layers::Linear linear(2, 2);
    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 1.f}, true);
    auto out = linear.forward(x);

    dfml::Tensor<float> dr({1, 2}, std::vector<float>{1.f, 1.f});
    out.accumulate_grad(dr);
    out.backward();

    check(linear.weights().has_grad(), "linear: weights should have grad before zero_grad");
    linear.zero_grad();
    for (size_t i = 0; i < linear.weights().nr_elements(); ++i)
        check_close(linear.weights().grad()[i], 0.f,
            "linear zero_grad: weights grad[" + std::to_string(i) + "] must be zero");
    for (size_t i = 0; i < linear.biases().nr_elements(); ++i)
        check_close(linear.biases().grad()[i], 0.f,
            "linear zero_grad: biases grad[" + std::to_string(i) + "] must be zero");
}

// ── Activation layers ─────────────────────────────────────────────────────────

void test_relu_layer_output_shape() {
    dfml::layers::ReLU relu;
    dfml::Tensor<float> x({2, 3}, std::vector<float>(6, 1.f));
    auto out = relu.forward(x);
    check(out.size(0) == 2 && out.size(1) == 3, "relu layer must preserve shape");
}

void test_relu_layer_clamps_negatives() {
    dfml::layers::ReLU relu;
    dfml::Tensor<float> x({1, 4}, std::vector<float>{-2.f, -1.f, 0.f, 1.f});
    auto out = relu.forward(x);
    check_close(out[0], 0.f, "relu layer: negative clamped [0]");
    check_close(out[1], 0.f, "relu layer: negative clamped [1]");
    check_close(out[2], 0.f, "relu layer: zero stays zero [2]");
    check_close(out[3], 1.f, "relu layer: positive unchanged [3]");
}

void test_sigmoid_layer_output_range() {
    dfml::layers::Sigmoid sigmoid;
    dfml::Tensor<float> x({1, 3}, std::vector<float>{-10.f, 0.f, 10.f});
    auto out = sigmoid.forward(x);
    for (size_t i = 0; i < 3; ++i)
        check(out[i] > 0.f && out[i] < 1.f, "sigmoid layer output must be in (0,1)");
}

void test_tanh_layer_output_range() {
    dfml::layers::Tanh tanh;
    dfml::Tensor<float> x({1, 3}, std::vector<float>{-10.f, 0.f, 10.f});
    auto out = tanh.forward(x);
    for (size_t i = 0; i < 3; ++i)
        check(out[i] >= -1.f && out[i] <= 1.f, "tanh layer output must be in [-1,1]");
}

void test_softmax_layer_sums_to_one() {
    dfml::layers::Softmax softmax;
    dfml::Tensor<float> x({2, 3}, std::vector<float>{1.f, 2.f, 3.f, 1.f, 1.f, 1.f});
    auto out = softmax.forward(x);
    float row0 = out[0] + out[1] + out[2];
    float row1 = out[3] + out[4] + out[5];
    check_close(row0, 1.f, "softmax layer: row 0 must sum to 1");
    check_close(row1, 1.f, "softmax layer: row 1 must sum to 1");
}

void test_activation_layers_have_no_parameters() {
    dfml::layers::ReLU    relu;
    dfml::layers::Sigmoid sigmoid;
    dfml::layers::Tanh    tanh;
    dfml::layers::Softmax softmax;
    check(relu.parameters().empty(),    "relu: parameters() must be empty");
    check(sigmoid.parameters().empty(), "sigmoid: parameters() must be empty");
    check(tanh.parameters().empty(),    "tanh: parameters() must be empty");
    check(softmax.parameters().empty(), "softmax: parameters() must be empty");
}

// ── Sequential ────────────────────────────────────────────────────────────────

void test_sequential_output_shape() {
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(4, 3)
         .add<dfml::layers::ReLU>()
         .add<dfml::layers::Linear>(3, 2);

    dfml::Tensor<float> x({5, 4}, std::vector<float>(20, 0.5f));
    auto out = model.forward(x);

    check(out.nr_dimensions() == 2, "sequential output must be 2D");
    check(out.size(0) == 5,         "sequential output rows must match batch size");
    check(out.size(1) == 2,         "sequential output cols must match last layer out_features");
}

void test_sequential_parameters_count() {
    // Two Linear layers: each has weights + biases → 4 total
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(3, 4)
         .add<dfml::layers::ReLU>()
         .add<dfml::layers::Linear>(4, 2);

    check(model.parameters().size() == 4,
          "sequential with 2 linear layers must have 4 parameters");
}

void test_sequential_known_values() {
    // Linear(2,2) identity weights + zero biases, then ReLU (all positive)
    // x = [[1,2]] → linear out = [[1,2]] → relu out = [[1,2]]
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(2, 2)
         .add<dfml::layers::ReLU>();

    auto params = model.parameters();
    // params[0] = weights {2,2}, params[1] = biases {2}
    float* w = params[0].data();
    w[0]=1.f; w[1]=0.f;
    w[2]=0.f; w[3]=1.f;
    params[1].zero();

    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 2.f});
    auto out = model.forward(x);

    check_close(out[0], 1.f, "sequential known: out[0]");
    check_close(out[1], 2.f, "sequential known: out[1]");
}

void test_sequential_relu_clamps_negatives() {
    // Linear(2,1) with weights [[-1]], bias [0]
    // x = [[1,0]] → output = [[-1]] → relu → [[0]]
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(2, 1)
         .add<dfml::layers::ReLU>();

    auto params = model.parameters();
    float* w = params[0].data();
    w[0]=-1.f; w[1]=0.f;
    params[1].zero();

    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 0.f});
    auto out = model.forward(x);

    check_close(out[0], 0.f, "sequential relu clamp: negative linear output must be zeroed");
}

void test_sequential_backward() {
    // Linear(2,1) identity-ish, then check gradients flow
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(2, 1);

    auto params = model.parameters();
    float* w = params[0].data();
    w[0]=1.f; w[1]=1.f;
    params[1].zero();

    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 2.f}, true);
    auto out = model.forward(x);

    dfml::Tensor<float> dr({1, 1}, std::vector<float>{1.f});
    out.accumulate_grad(dr);
    out.backward();

    check(x.has_grad(),       "sequential backward: x must have grad");
    check_close(x.grad()[0], 1.f, "sequential backward: dx[0]");
    check_close(x.grad()[1], 1.f, "sequential backward: dx[1]");
}

void test_sequential_zero_grad() {
    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(2, 2)
         .add<dfml::layers::Linear>(2, 1);

    dfml::Tensor<float> x({1, 2}, std::vector<float>{1.f, 1.f}, true);
    auto out = model.forward(x);

    dfml::Tensor<float> dr({1, 1}, std::vector<float>{1.f});
    out.accumulate_grad(dr);
    out.backward();

    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i)
        check(params[i].has_grad(),
              "sequential zero_grad: param[" + std::to_string(i) + "] should have grad before reset");

    model.zero_grad();

    params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i)
        for (size_t j = 0; j < params[i].nr_elements(); ++j)
            check_close(params[i].grad()[j], 0.f,
                "sequential zero_grad: param[" + std::to_string(i) + "] grad[" + std::to_string(j) + "] must be zero");
}

}  // namespace

int main() {
    try {
        // Linear
        test_linear_output_shape();
        test_linear_parameters_count();
        test_linear_biases_initialized_to_zero();
        test_linear_forward_known_values();
        test_linear_forward_with_bias();
        test_linear_backward();
        test_linear_zero_grad();

        // Activation layers
        test_relu_layer_output_shape();
        test_relu_layer_clamps_negatives();
        test_sigmoid_layer_output_range();
        test_tanh_layer_output_range();
        test_softmax_layer_sums_to_one();
        test_activation_layers_have_no_parameters();

        // Sequential
        test_sequential_output_shape();
        test_sequential_parameters_count();
        test_sequential_known_values();
        test_sequential_relu_clamps_negatives();
        test_sequential_backward();
        test_sequential_zero_grad();

        std::cout << "All layers tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Layers tests failed: " << e.what() << '\n';
        return 1;
    }
}
