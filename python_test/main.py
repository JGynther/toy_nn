import numpy as np
from math import pi
import matplotlib.pyplot as plt


class MSE:
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, x):
        self.input = x
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, x, learning_rate):
        input_error = np.dot(x, self.weights.T)
        weights_error = np.dot(self.input.T, x)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(x)

        return input_error


class Activation:
    def __init__(self, function):
        self.activation = function.activation
        self.activation_prime = function.gradient

    def forward(self, x):
        self.input = x
        return self.activation(x)

    def backward(self, x, _learning_rate):
        return self.activation_prime(self.input) * x


class ReLU(Activation):
    def __init__(self):
        super().__init__(self)

    def activation(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return x > 0


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def loss_fn(self, loss_fn):
        self.loss = loss_fn.loss
        self.loss_prime = loss_fn.gradient

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, error, learning_rate):
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)


model = Sequential(
    Dense(1, 128),
    ReLU(),
    Dense(128, 128),
    ReLU(),
    Dense(128, 1),
)

model.loss_fn(MSE)

epochs = 30_000
learning_rate = 0.001

# Train model
for i in range(epochs):
    batch = np.random.rand(16, 1) * pi * 2
    labels = np.sin(batch)

    output = model(batch)
    error = model.loss_prime(labels, output)

    model.backward(error, learning_rate)

    if i % 100 == 99:
        loss = model.loss(labels, output)
        print(f"Epoch {i}, error {loss}")


# Plot model predictions
x = np.array([np.linspace(0, pi * 2, 1000)]).T
y = np.sin(x)
output = model(x)

plt.plot(x, y, color="blue")
plt.plot(x, output, color="red")
plt.show()
