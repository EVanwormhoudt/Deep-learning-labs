import numpy as np
from .Layers import Hiddenlayer, InputLayer
from dataclasses import dataclass


@dataclass
class MultiLayerPerceptron:
    layers: list[Hiddenlayer] = None
    input_layer: InputLayer = None

    def __post_init__(self) -> None:
        """Initialize the MLP."""
        self.layers = []

    def add_layer(self, neurons: int, activation: str) -> None:
        """Add a layer to the MLP."""
        self.layers.append(Hiddenlayer(neurons, activation))

        if len(self.layers) > 1:
            self.layers[-1].previous_layer = self.layers[-2]
            self.layers[-2].next_layer = self.layers[-1]
        else:
            self.layers[-1].previous_layer = self.input_layer
        self.layers[-1].generate_weights()

    def add_input_layer(self, inputs: int) -> None:
        """Add a layer to the MLP."""
        self.input_layer = InputLayer(inputs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the MLP."""
        self.input_layer.features = features = x

        for layer in self.layers:
            features = layer.predict(features)
        return features

    def back_propagate(self, error: np.ndarray) -> None:
        """Back propagate the error."""
        self.layers[-1].delta = error * self.layers[-1].activation_function(self.layers[-1].sigma, derivative=True)
        for layer in reversed(self.layers[:-1]):
            layer.back_propagate()

    def forward_propagate(self, learning_rate: float, sigma: np.ndarray) -> None:
        """Forward propagate the error."""
        for layer in self.layers:
            layer.forward_propagate(learning_rate, sigma)
            sigma = layer.sigma

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate) -> None:
        """Train the MLP."""
        for epoch in range(epochs):
            predictions = []
            for x_i, y_i in zip(x, y):
                prediction = self.predict(x_i)
                predictions.append(prediction)
                error = y_i - prediction
                self.back_propagate(error)
                self.forward_propagate(learning_rate, self.input_layer.features)

            # print(f"Epoch: {epoch}, MSE: {self.MSE(self.layers[-1].sigma, y)}")
            print(f" MSE: {self.MSE(np.array(np.around(predictions)), y)}")

    def MSE(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Calculate the mean squared error."""
        return np.average((y - predictions) ** 2)
