import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
import matplotlib.pyplot as plt


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    n = 10
    index = np.random.randint(len(X_train), size=n)
    plt.figure(figsize=(n*1.5, 1.5))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        ax.set_title('{} ({})'.format(y_train[index[i]],index[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(X_train[index[i]])
    plt.show()

