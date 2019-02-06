import cv2
import numpy as np
from lines import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from helpers import *


def load_trained_model(path):

    loaded_model = tf.keras.models.load_model(path)

    return loaded_model


def save_trained_model(path, model):

    model.save('digit_trained.model')


def train_cnn():

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = tf.keras.models.Sequential()

    x_train = tf.keras.utils.normalize(x_train, axis=1)

    model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100)

    save_trained_model('trained.model', model)

    return model


def predict_cnn(model, data):

    predictions = model.predict(data)
    print(np.argmax(predictions[3]))
    plt.imshow(data[3], cmap=plt.cm.binary)

    plt.show()


def invert_data(data):

    inverted = []
    for d in data:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona

        inverted_one = erode(erode(invert(d)))
        inverted.append(inverted_one)

    return np.array(inverted)

