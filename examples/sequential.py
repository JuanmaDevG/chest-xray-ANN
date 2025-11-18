# Trying the sequential api

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    my_model = keras.Sequential([
        layers.Input(shape=(28,28)),
        layers.Rescaling(scale=1./255),
        layers.Flatten(),
        layers.Dense(256),
        layers.Dense(256),
        layers.Dense(10, activation='softmax')
        ])
    
    my_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    my_model.fit(X_train, y_train, epochs=5, verbose=1)
