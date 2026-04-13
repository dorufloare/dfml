#include <iostream>
#include "dfml/dfml.hpp"

void train_seq_xor(int nr_epochs=5000) {
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
	model.add<dfml::layers::ReLU>();
	model.add<dfml::layers::Linear>(4, 1);
	model.add<dfml::layers::Sigmoid>();

	dfml::optim::SGD optimizer(model.parameters(), 0.1f);
	for (int epoch = 0; epoch < nr_epochs; ++epoch) {
		auto prediction = model.forward(X);
		auto loss = dfml::ops::mse_loss(prediction, Y);
		loss.backward();
		optimizer.step();
		optimizer.zero_grad();
		if (epoch % 500 == 0) {
            std::cout << "epoch " << epoch
                      << "  loss: " << loss.data()[0]
                      << "\n";
        }
	}

	std::cout << "\nFinal predictions:\n";
    dfml::GradGuard guard(true);  
    auto pred = model.forward(X);
 
    std::cout << "[0,0] -> " << pred.data()[0] << "  (expected 0)\n";
    std::cout << "[0,1] -> " << pred.data()[1] << "  (expected 1)\n";
    std::cout << "[1,0] -> " << pred.data()[2] << "  (expected 1)\n";
    std::cout << "[1,1] -> " << pred.data()[3] << "  (expected 0)\n";
}

int main() {
	train_seq_xor();

	return 0;
}
