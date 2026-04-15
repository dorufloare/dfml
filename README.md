# dfml

A header-only C++ deep learning library built from scratch. No dependencies beyond the standard library.

The goal is a clean, readable implementation of the core pieces: tensors with autograd, common layers, optimizers, and a training loop. Everything lives in headers so there is nothing to link against.

If you have used PyTorch, the API will feel familiar. The mental model is the same: build a model, define a loss, call backward, step the optimizer. The main difference is that everything is explicit C++ rather than Python with a C++ backend.

## About the library

Autograd works by recording a computation graph as tensors flow through operations. Each tensor stores its parent tensors and a backward function. Calling `.backward()` on a scalar loss does a reverse topological traversal and accumulates gradients.

Very roughly, a training step does this:
1. Forward pass: inputs flow through layers, each op records its backward function.
2. Compute loss from predictions and targets.
3. `loss.backward()` walks the graph in reverse, calling each backward function.
4. Optimizer reads accumulated gradients and updates parameters.
5. Zero gradients before the next step.

`Trainer` wraps this loop. `GradGuard` disables graph recording during inference so no memory is wasted on backward hooks.

## What this project does

This is a C++ neural network library with autograd, feedforward layers, two optimizers, two loss functions, and a training utility. `src/main.cpp` has three ready-to-run demos that cover classification and regression:

1. **XOR** — learns the XOR gate with a tiny 2-4-1 network and SGD.
2. **Circle** — binary classification of 2D points inside a circle, with train/test split and accuracy reporting.
3. **Function approximation** — fits a piecewise-discontinuous function over `[-10, 10]` with a deeper network and Adam.

Run them all with `./run.sh`.

## Quick start

```bash
./run.sh
```

Run tests:

```bash
./test_run.sh
```

## PyTorch comparison

The core pattern is the same. Here is the same two-layer network written in both:

**PyTorch**
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
```

**dfml**
```cpp
#include "dfml/dfml.hpp"

dfml::layers::Sequential model;
model.add<dfml::layers::Linear>(2, 16);
model.add<dfml::layers::Tanh>();
model.add<dfml::layers::Linear>(16, 1);
model.add<dfml::layers::Sigmoid>();

dfml::optim::Adam optimizer(model.parameters());
dfml::ops::LossFn loss_fn(dfml::ops::mse_loss<float>);

dfml::Trainer trainer(model, optimizer, loss_fn);
trainer.fit(X, Y, /*epochs=*/1000, /*print_every=*/100);
```

The concepts map directly: `nn.Sequential` → `dfml::layers::Sequential`, `nn.Linear` → `dfml::layers::Linear`, `torch.optim.Adam` → `dfml::optim::Adam`. The main difference is that dfml's `Trainer` bundles the epoch loop so you do not have to write it yourself.

For a raw loop without `Trainer`, the structure is the same as PyTorch:

```cpp
for (size_t epoch = 1; epoch <= epochs; ++epoch) {
    optimizer.zero_grad();
    auto pred = model.forward(X);
    auto loss = loss_fn(pred, Y);
    loss.backward();
    optimizer.step();
}
```

## API reference

### Tensor

`dfml::Tensor<T>` is the core type. All operations on tensors with `requires_grad = true` build the autograd graph automatically.

```cpp
// construction
dfml::Tensor<float> t({3, 4});                        // shape only, uninitialized
dfml::Tensor<float> t({2, 2}, {1.f, 2.f, 3.f, 4.f}); // shape + data
auto s = dfml::Tensor<float>::scalar(1.f, true);      // scalar with grad

// access
t.shape();           // std::vector<size_t>
t.size(dim);         // size along one dimension
t.nr_elements();     // total element count
t[i];                // flat index
t.at({row, col});    // multi-dimensional index
t.data();            // raw pointer

// views and copies
t.view({6, 1});      // reshape (no copy)
t.clone();           // deep copy

// grad
t.requires_grad();
t.grad();            // gradient tensor
t.zero_grad();
t.backward();        // start reverse pass from this scalar
```

### Layers

All layers inherit from `dfml::layers::Layer` and implement `forward` and `parameters`.

| Layer | PyTorch equivalent | Description |
|---|---|---|
| `Linear(in, out)` | `nn.Linear(in, out)` | Fully connected: `x @ W + b`, Xavier-normal init |
| `ReLU` | `nn.ReLU()` | `max(0, x)` elementwise |
| `Sigmoid` | `nn.Sigmoid()` | `1 / (1 + exp(-x))` elementwise |
| `Tanh` | `nn.Tanh()` | `tanh(x)` elementwise |
| `Softmax` | `nn.Softmax(dim=-1)` | Row-wise softmax |

Build a network with `Sequential`:

```cpp
dfml::layers::Sequential model;
model.add<dfml::layers::Linear>(2, 16);
model.add<dfml::layers::Tanh>();
model.add<dfml::layers::Linear>(16, 1);
model.add<dfml::layers::Sigmoid>();

auto output = model.forward(input);
auto params = model.parameters(); // flat list of all weight/bias tensors
```

### Optimizers

| Optimizer | PyTorch equivalent | Constructor defaults |
|---|---|---|
| `SGD(params, lr=0.1)` | `torch.optim.SGD(params, lr=0.1)` | Vanilla gradient descent |
| `Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)` | `torch.optim.Adam(params)` | Adaptive moment estimation |

```cpp
dfml::optim::Adam optimizer(model.parameters());
optimizer.zero_grad();
// ... forward + backward ...
optimizer.step();
```

### Loss functions

Both functions return a scalar tensor with the backward function attached.

```cpp
// mean squared error  (PyTorch: nn.MSELoss())
dfml::ops::mse_loss<float>(prediction, target);

// cross-entropy from logits  (PyTorch: nn.CrossEntropyLoss())
dfml::ops::cross_entropy_loss<float>(logits, labels);    // labels: std::vector<size_t>
dfml::ops::cross_entropy_loss<float>(logits, {0, 1, 2}); // initializer_list
dfml::ops::cross_entropy_loss<float>(logits, label_tensor); // Tensor<float> of class indices
```

Wrap a loss function for use with `Trainer`:

```cpp
dfml::ops::LossFn loss_fn(dfml::ops::mse_loss<float>);
```

### Trainer

`Trainer` handles the epoch loop, gradient zeroing, and metric printing.

```cpp
dfml::Trainer trainer(model, optimizer, loss_fn);
trainer.add_metric("accuracy", dfml::binary_accuracy);

// full-batch training (batch_size=0 means full batch)
auto train_pred = trainer.fit(X, Y, /*epochs=*/2000, /*print_every=*/100);

// mini-batch training
auto train_pred = trainer.fit(X, Y, /*epochs=*/2000, /*batch_size=*/32, /*print_every=*/100);

// inference (no grad, no graph recorded)
auto test_pred = trainer.predict(X_test);
```

`fit` returns the final predictions on the training set.

### Data utilities

```cpp
// reproducible randomness
dfml::set_rng_seed(42);
std::mt19937& rng = dfml::global_rng();

// train/test split (preserves row order, no shuffle)
auto [X_train, X_test] = dfml::train_test_split(X, 0.8f);

// in-place shuffle of X and Y together (keeps rows aligned)
dfml::shuffle(X, Y);
```

### Metrics

```cpp
dfml::binary_accuracy(pred, target); // fraction of predictions where round(p) == round(t)
dfml::mse(pred, target);             // mean squared error
dfml::mae(pred, target);             // mean absolute error
```

### Weight initialization

Applied automatically by `Linear`. Available standalone if you build custom layers:

```cpp
dfml::init::xavier_uniform(tensor, fan_in, fan_out);
dfml::init::xavier_normal(tensor, fan_in, fan_out);
dfml::init::kaiming_normal(tensor, fan_in);
dfml::init::zeros(tensor);
```

### GradGuard

Disables gradient tracking for any code in its scope. Used internally by `Trainer::predict` and the final prediction pass in `Trainer::fit`. Equivalent to PyTorch's `torch.no_grad()`.

```cpp
{
    dfml::GradGuard guard;
    auto out = model.forward(X); // no graph built, no backward hooks
}
```

## Sample output

These are real numbers from running `./run.sh` with seed 42.

```text
=== XOR ===
epoch 500   loss: 0.15058
epoch 1000  loss: 0.035561
epoch 1500  loss: 0.013887
epoch 2000  loss: 0.007966
epoch 2500  loss: 0.005437
epoch 3000  loss: 0.004077
epoch 3500  loss: 0.003239
epoch 4000  loss: 0.002676
epoch 4500  loss: 0.002274
epoch 5000  loss: 0.001973
[0,0] -> 0.0208  (expected 0)
[0,1] -> 0.9522  (expected 1)
[1,0] -> 0.9529  (expected 1)
[1,1] -> 0.0544  (expected 0)

=== Circle ===
train: 160 examples
test:  40 examples

training...
epoch 100   loss: 0.239226  accuracy: 0.5875
epoch 200   loss: 0.231328  accuracy: 0.59375
epoch 300   loss: 0.193095  accuracy: 0.78125
epoch 400   loss: 0.144745  accuracy: 0.84375
epoch 500   loss: 0.101805  accuracy: 0.91875
epoch 600   loss: 0.068916  accuracy: 0.96875
epoch 700   loss: 0.048382  accuracy: 0.975
epoch 800   loss: 0.036913  accuracy: 0.98125
epoch 900   loss: 0.029975  accuracy: 0.98125
epoch 1000  loss: 0.025299  accuracy: 0.98125
epoch 1700  loss: 0.012041  accuracy: 1.0
epoch 2000  loss: 0.009488  accuracy: 1.0

train accuracy: 100%
test accuracy:  95%

=== Function approximation ===
epoch 500   loss: 108.259  mse: 108.259  mae: 4.725
epoch 1000  loss:  60.984  mse:  60.984  mae: 3.208
epoch 1500  loss:  35.005  mse:  35.005  mae: 2.211
epoch 2000  loss:  20.105  mse:  20.105  mae: 1.536
epoch 2500  loss:  11.605  mse:  11.605  mae: 1.102
epoch 3000  loss:   6.596  mse:   6.596  mae: 0.812
epoch 3500  loss:   3.742  mse:   3.742  mae: 0.575
epoch 4000  loss:   2.233  mse:   2.233  mae: 0.452
epoch 4500  loss:   1.362  mse:   1.362  mae: 0.390
epoch 5000  loss:   0.775  mse:   0.775  mae: 0.302

train mse: 0.775
test mse:  0.800
pred: 0.854  actual: 1.000  diff: 0.145
pred: -0.133 actual: 0.058  diff: 0.191
pred: 28.257 actual: 28.398 diff: 0.140
pred: 27.911 actual: 28.097 diff: 0.186
pred: 1.591  actual: 1.983  diff: 0.392
pred: 0.241  actual: 0.000  diff: 0.241
pred: 17.580 actual: 17.402 diff: 0.177
pred: 7.151  actual: 7.177  diff: 0.025
pred: 19.216 actual: 19.448 diff: 0.231
pred: 0.039  actual: 0.000  diff: 0.039
```

XOR converges cleanly. The circle demo hits 100% train accuracy and 95% test with 160 examples. The function approximation is the harder task — the target is piecewise-discontinuous with jumps, so some residual error near the boundaries is expected.

## File map

### Top level
- `CMakeLists.txt` — build setup, C++20 required
- `run.sh` — configure, build, and run the main executable
- `test_run.sh` — configure, build, and run tests via CTest

### include/dfml
- `dfml.hpp` — single include for everything
- `tensor.hpp` — `Tensor<T>` definition and backward traversal
- `trainer.hpp` — `Trainer` class with fit/predict/metrics

### include/dfml/layers
- `layer.hpp` — abstract `Layer` base class
- `linear.hpp` — `Linear` layer
- `activation.hpp` — `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `sequential.hpp` — `Sequential` container

### include/dfml/optim
- `optimizer.hpp` — abstract `Optimizer` base class
- `sgd.hpp` — SGD
- `adam.hpp` — Adam

### include/dfml/ops
- `ops.hpp` — umbrella include
- `add.hpp` — elementwise add and bias broadcast
- `matrix_multiply.hpp` — matmul with backward
- `matrix_transpose.hpp` — transpose with backward
- `activation/` — relu, sigmoid, tanh, softmax ops with backward
- `loss/loss_fn.hpp` — `LossFn` type alias
- `loss/mse_loss.hpp` — MSE with backward
- `loss/cross_entropy_loss.hpp` — cross-entropy from logits with backward

### include/dfml/init
- `init.hpp` — Xavier uniform/normal, Kaiming normal, zeros

### include/dfml/utils
- `random.hpp` — `global_rng()`, `set_rng_seed()`
- `data.hpp` — `train_test_split`, `shuffle`
- `metrics.hpp` — `binary_accuracy`, `mse`, `mae`

### include/dfml/autograd
- `autograd_metadata.hpp` — backward function and parent storage inside `TensorImpl`
- `tensor_autograd.hpp` — operator overloads that hook into the graph
- `grad_guard.hpp` — `GradGuard` RAII scope

### include/dfml/internal
- `tensor_impl.hpp` — `TensorImpl<T>`, the shared backing for `Tensor`
- `storage.hpp` — flat data buffer

### src
- `src/main.cpp` — three ready-to-run demos: XOR, circle classification, function approximation

### tests
- `tests/` — tensor correctness tests, run via `test_run.sh`

## Design notes

The library uses `shared_ptr<TensorImpl>` so tensors are cheap to copy and the graph naturally keeps parents alive until backward is done. `GradGuard` is a thread-local flag so inference is zero-overhead with no API change.

The tradeoff right now is expressiveness vs complexity. There is no support for custom backward functions from user code, no GPU path, and no dynamic shapes. The abstractions are sized for the current feature set, not for hypothetical future ones.
