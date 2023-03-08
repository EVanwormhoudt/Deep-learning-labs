from dataclasses import dataclass
import numpy as np
from .activation_functions import sigmoid, tanh, relu, softmax


@dataclass
class Hiddenlayer:
    neurons: int
    activation: str

    weights: np.ndarray = None
    bias: np.ndarray = None
    delta: np.ndarray = None
    sigma: np.ndarray = None

    previous_layer = None
    next_layer = None

    activation_function: callable = None

    def __post_init__(self):

        match self.activation:
            case "sigmoid":
                self.activation_function = sigmoid
            case "tanh":
                self.activation_function = tanh
            case "relu":
                self.activation_function = relu
            case "softmax":
                self.activation_function = softmax

    def generate_weights(self):
        self.weights = np.random.uniform(-1, 1, size=(self.neurons, self.previous_layer.neurons))
        self.bias = np.random.uniform(-1, 1, size=(self.neurons, 1))

        self.delta = np.zeros(self.bias.shape)
        self.sigma = np.zeros(self.bias.shape)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.sigma = self.activation_function(np.add(np.matmul(self.weights, x).reshape(-1, 1), self.bias))
        return self.sigma

    def back_propagate(self) -> None:
        weights = self.next_layer.weights
        delta = self.next_layer.delta
        delta_j_x_weight = np.matmul(weights.T, delta)
        self.delta = delta_j_x_weight * self.activation_function(self.sigma, derivative=True)

    def forward_propagate(self, learning_rate: float, sigma: np.ndarray) -> None:
        for row in range(len(self.weights)):
            for col in range(len(self.weights[0])):
                self.weights[row][col] += learning_rate * self.delta[row] * sigma[col]
        self.bias += self.delta * learning_rate


@dataclass
class InputLayer:
    neurons: int
    features: np.ndarray = None

    def __post_init__(self):
        self.features = np.zeros((self.neurons, 1))

    def predict(self, x):
        self.features = x
        return self.features

    def back_propagate(self, delta, weights):
        pass

    def forward_propagate(self, learning_rate, sigma):
        pass
