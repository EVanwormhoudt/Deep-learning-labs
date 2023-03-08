from datetime import datetime

from tensorflow.keras.layers import Dense
from keras.models import Sequential
import numpy as np

from tensorflow import keras

values = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

for i in range(5):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_squared_error", "mean_absolute_error"])

    print("Model #", i + 1)
    model.fit(values, labels, epochs=100, batch_size=2, callbacks=[tensorboard_callback])
    scores = model.evaluate(values, labels)

    print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print(scores)
    print(model.predict(values).round())
    print(model.weights)
