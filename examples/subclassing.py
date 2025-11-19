# Trying subclassing with keras.Model parent class to build a model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Model(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(Model,self).__init__( **kwargs)
        self.input_layer = keras.layers.Input(input_shape)
        self.rescaling = layers.Rescaling(scale=1./255)
        self.flatten = layers.Flatten()
        self.v_dense = [layers.Dense(256), layers.Dense(256)]
        self.out_dense = layers.Dense(10, activation='softmax')
        
        x = self.rescaling(self.input_layer)
        x = self.flatten(x)
        x = self.v_dense[0](x)
        x = self.v_dense[1](x)
        self.output_layer = self.out_dense(x)

        super().__init__(inputs=self.input_layer,outputs=self.output_layer, **kwargs)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    model = Model((28,28))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=1)
