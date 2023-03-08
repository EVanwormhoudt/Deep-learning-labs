import copy
from dataclasses import dataclass
import numpy as np

values = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels = [0, 0, 0, 1]

weights = [[-1, 1], [-0.8, 0.1]]
biases = [0.5, 0]

learning_rate = [0.002, 0.02, 0.2, 2]


@dataclass
class Perceptron:
    weights: list[float]
    bias: float
    learning_rate: float

    def train(self, inputs: list[tuple[int, int]], label: list[int],epoch:int) -> None:
        """Train the perceptron for n iteration."""
        predictions = []
        iteration = 0
        while not np.array_equal(predictions,label):
            predictions = []
            for i,input in enumerate(inputs):
                prediction = self.predict(input)
                predictions.append(prediction)
                error = label[i] - prediction
                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, input)]
                self.bias += self.learning_rate * error
            iteration += 1
            if iteration >= epoch:
                print("didn't converge")
                return
        print(f"Converged in {iteration} epochs")

    def predict(self, inputs: tuple[int, int]) -> int:
        """Predict the output for a given input."""
        result = (sum([w * x for w, x in zip(self.weights, inputs)]) + self.bias > 0)
        return result > 0

    def test(self, inputs: tuple[int, int]) -> None:
        """Test the perceptron."""
        prediction = self.predict(inputs)
        print(f"Input: {inputs}, Prediction: {prediction}")

    def __str__(self):
        weights = [round(w, 2) for w in self.weights]
        bias = round(self.bias, 2)
        return f"weights: {weights}, bias: {bias}"

    def compare(self,predictions,labels):
        return predictions == labels



for weight in weights:
    for bias in biases:
        for lr in learning_rate:
            weight = copy.deepcopy(weight)
            print("=========================================")
            print("Weights: ", weight, "Bias: ", bias, "Learning Rate: ", lr)
            perceptron = Perceptron(weight, bias, lr)
            perceptron.train(values, labels,10000)
            #perceptron.test((1,1))
            print(perceptron.__str__())

