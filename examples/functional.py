# Trying the functional api

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    ann_input = layers.Input(shape=(28,28))
    x = layers.Rescaling(scale=1./255)(ann_input)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(256)(x)

    ann_output = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(ann_input, ann_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=1)
