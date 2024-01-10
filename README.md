# toy_nn

A naive neural network implementation from scratch in Rust for fun. Also includes a roughly equivalent [Python implementation](https://github.com/JGynther/toy_nn/blob/main/python_test/main.py) for testing purposes.

You can see an example of fitting a dense/linear NN to a sine wave in the [main.rs](https://github.com/JGynther/toy_nn/blob/main/src/main.rs) file.

```rust
// Create a NN
let mut model = Sequential::new(MSE {});

model.add(Dense::new(2, 3));
model.add(ReLU::new());
model.add(Dense::new(3, 1));
```

```rust
// Predict
let pred = model.forward(&x);

// Backprop
let error = model.loss_fn.loss_prime(&pred, &y);
model.backward(&error, learning_rate);
```
