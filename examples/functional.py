# Trying the functional api

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
