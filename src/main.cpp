#include <iostream>
#include "dfml/dfml.hpp"

void train_xor() {
	dfml::Tensor<float> X({4, 2}, {
        0.f, 0.f,
        0.f, 1.f,
        1.f, 0.f,
        1.f, 1.f
    });

    dfml::Tensor<float> Y({4, 1}, {
        0.f,
        1.f,
        1.f,
        0.f
    });

	dfml::layers::Sequential model;
	model.add<dfml::layers::Linear>(2, 4);
	model.add<dfml::layers::Tanh>();
	model.add<dfml::layers::Linear>(4, 1);
	model.add<dfml::layers::Sigmoid>();

	dfml::optim::SGD optimizer(model.parameters());
	dfml::ops::LossFn loss(dfml::ops::mse_loss<float>);

    dfml::Trainer trainer(model, optimizer, loss);
    auto d = trainer.fit(X, Y, 5000, 500);
    std::cout << "[0,0] -> " << d[0] << "  (expected 0)\n";
    std::cout << "[0,1] -> " << d[1] << "  (expected 1)\n";
    std::cout << "[1,0] -> " << d[2] << "  (expected 1)\n";
    std::cout << "[1,1] -> " << d[3] << "  (expected 0)\n";
}

void train_circle() {
    const size_t N = 200;
    const float radius = 0.7f;

    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> X_data, Y_data;
    X_data.reserve(N * 2);
    Y_data.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        float x = dist(dfml::global_rng());
        float y = dist(dfml::global_rng());
        X_data.push_back(x);
        X_data.push_back(y);
        Y_data.push_back(x * x + y * y < radius * radius ? 1.f : 0.f);
    }
    dfml::Tensor<float> X({N, 2}, X_data);
    dfml::Tensor<float> Y({N, 1}, Y_data);

    auto [X_train, X_test] = dfml::train_test_split(X, 0.8f);
    auto [Y_train, Y_test] = dfml::train_test_split(Y, 0.8f);

    std::cout << "train: " << X_train.size(0) << " examples\n";
    std::cout << "test:  " << X_test.size(0)  << " examples\n\n";

    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(2, 16);
    model.add<dfml::layers::Tanh>();
    model.add<dfml::layers::Linear>(16, 8);
    model.add<dfml::layers::Tanh>();
    model.add<dfml::layers::Linear>(8, 1);
    model.add<dfml::layers::Sigmoid>();

    dfml::optim::Adam optimizer(model.parameters());
    dfml::ops::LossFn loss_fn(dfml::ops::mse_loss<float>);

    dfml::Trainer trainer(model, optimizer, loss_fn);
    trainer.add_metric("accuracy", dfml::binary_accuracy);

    std::cout << "training...\n";
    auto train_pred = trainer.fit(X_train, Y_train, 2000, 100);

    auto test_pred = trainer.predict(X_test);

    float train_acc = dfml::binary_accuracy(train_pred, Y_train);
    float test_acc  = dfml::binary_accuracy(test_pred,  Y_test);

    std::cout << "\ntrain accuracy: " << train_acc * 100.f << "%\n";
    std::cout << "test accuracy:  " << test_acc  * 100.f << "%\n";
}

void train_fn(std::function<float(float)> fn) {
    std::vector<float> x_data, y_data;
    std::uniform_real_distribution<float> dist(-10,10);

    const int N = 5000;
    for (int i = 0; i < N; ++i) {
        x_data.push_back(dist(dfml::global_rng()));
        y_data.push_back(fn(x_data.back()));
    }

    dfml::Tensor<float> X({N, 1}, x_data);
    dfml::Tensor<float> y({N, 1}, y_data);

    const float split = 0.8;
    auto [X_train, X_test] = dfml::train_test_split(X, split);
    auto [Y_train, Y_test] = dfml::train_test_split(y, split);

    dfml::layers::Sequential model;
    model.add<dfml::layers::Linear>(1, 64);
    model.add<dfml::layers::Tanh>();
    model.add<dfml::layers::Linear>(64, 32);
    model.add<dfml::layers::ReLU>();
    model.add<dfml::layers::Linear>(32, 16);
    model.add<dfml::layers::Tanh>();
    model.add<dfml::layers::Linear>(16, 1);

    dfml::optim::Adam optimizer(model.parameters());
    dfml::ops::LossFn loss_fn(dfml::ops::mse_loss<float>);

    dfml::Trainer trainer(model, optimizer, loss_fn);
    trainer.add_metric("mse", dfml::mse);
    trainer.add_metric("mae", dfml::mae);

    auto train_pred = trainer.fit(X_train, Y_train, 5000, 0, 500);

    auto test_pred = trainer.predict(X_test);

    float train_mse = dfml::mse(train_pred, Y_train);
    float test_mse  = dfml::mse(test_pred,  Y_test);

    std::cout << "\ntrain mse: " << train_mse << "\n";
    std::cout << "test mse:  " << test_mse << "\n";

    size_t n = test_pred.nr_elements();
    for (size_t i = n-10; i < n; ++i) {
        std::cout << "pred: " << test_pred[i] << " actual: " << Y_test[i] << " diff: " << std::fabs(test_pred[i] - Y_test[i]) << '\n';
    }
}


int main() {
    dfml::set_rng_seed(42);

    std::cout << "=== XOR ===\n";
    train_xor();

    std::cout << "\n=== Circle ===\n";
    train_circle();

    std::cout << "\n=== Function approximation ===\n";
    std::function<float(float)> complex_fn = [](float x) {
        if (x < -2) {
            return std::sin(5 * x) + 0.5f * x * x;
        } else if (x < -1) {
            return std::exp(-x * x) + std::fabs(x);
        } else if (x < 0) {
            return 1.0f / (1.0f + std::exp(-3 * x));
        } else if (x < 1) {
            return x * x * x - x + 2.0f;
        } else if (x < 2) {
            return std::cos(7 * x) + 1.0f / (x + 1.0f);
        } else if (x < 3) {
            float s = std::sin(3 * x);
            return (s > 0) ? 1.0f : (s < 0) ? -1.0f : 0.0f;
        } else if (x < 4) {
            return std::sqrt(std::fabs(x - 3.0f)) + std::sin(x * x);
        } else {
            return 0.0f;
        }
    };
    train_fn(complex_fn);

	return 0;
}
