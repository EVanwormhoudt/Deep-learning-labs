import copy

import numpy as np
from MyMLP.MultiLayerPerceptron import MultiLayerPerceptron


values = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
labels = np.array([0, 1, 1, 0])

learning_rate = [0.002, 0.02, 0.2, 2, 20]


model = MultiLayerPerceptron()
model.add_input_layer(2)
model.add_layer(8, "relu")
model.add_layer(4, "relu")
model.add_layer(2, "relu")
model.add_layer(1, "relu")
model.train(values, labels, 1000, 0.2)

