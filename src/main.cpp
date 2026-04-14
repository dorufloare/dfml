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

std::pair<dfml::Tensor<float>, dfml::Tensor<float>>
generate_circle(size_t n, float radius = 0.7f) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> X_data, Y_data;
    X_data.reserve(n * 2);
    Y_data.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        float x = dist(dfml::global_rng());
        float y = dist(dfml::global_rng());
        X_data.push_back(x);
        X_data.push_back(y);
        Y_data.push_back(x * x + y * y < radius * radius ? 1.f : 0.f);
    }

    dfml::Tensor<float> X({n, 2}, X_data);
    dfml::Tensor<float> Y({n, 1}, Y_data);
    return {X, Y};
}

void train_circle() {
    const size_t N = 200;

    auto [X, Y] = generate_circle(N);
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

int main() {
    dfml::set_rng_seed(42);  
	train_circle();

	return 0;
}
