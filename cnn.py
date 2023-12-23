import numpy as np
import random
import streamlit as st
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def load_dataset():
    X_train = np.loadtxt('tests/cnn_test/input.csv', delimiter=',')
    Y_train = np.loadtxt('tests/cnn_test/labels.csv', delimiter=',')

    X_test = np.loadtxt('tests/cnn_test/input_test.csv', delimiter=',')
    Y_test = np.loadtxt('tests/cnn_test/labels_test.csv', delimiter=',')

    X_train = X_train.reshape(len(X_train), 100, 100, 3)
    Y_train = Y_train.reshape(len(Y_train), 1)

    X_test = X_test.reshape(len(X_test), 100, 100, 3)
    Y_test = Y_test.reshape(len(Y_test), 1)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Shape of X_train: ", X_train.shape)
    print("Shape of Y_train: ", Y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of Y_test: ", Y_test.shape)

    return X_train, Y_train, X_test, Y_test

def display_random_image(X_train, image_caption="Random Training Image", width=300):
    idx = random.randint(0, len(X_train))
    st.image(X_train[idx, :], caption=image_caption, use_column_width=False, width=width)
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, epochs=5, batch_size=64):
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

def evaluate_model(model, X_test, Y_test):
    return model.evaluate(X_test, Y_test)

def display_random_test_image(X_test, Y_test, model, image_caption="Random Test Image", width=300):
    idx2 = random.randint(0, len(Y_test))
    st.image(X_test[idx2, :], caption=image_caption, use_column_width=False, width=width)

    y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
    y_pred = y_pred > 0.5

    if y_pred == 0:
        pred = 'dog'
    else:
        pred = 'cat'

    st.write(f'Model prediction: {pred}')
