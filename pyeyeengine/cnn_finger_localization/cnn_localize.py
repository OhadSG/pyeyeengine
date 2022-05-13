import datetime

import time

import cv2
import csv

import numpy as np
import keras
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def prep_data():
    reader = csv.DictReader(open('example4 - Copy.csv', 'r'))
    X = []
    Y = []
    for idx, line in enumerate(reader):
        if idx > 0:
           X.append(np.expand_dims(cv2.imread(line["img_path"]), axis=0))
           # print(X[-1].shape)
           Y.append(np.array([line["x"], line["y"]]).reshape(1, -1))

    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def create_model(input_shape, kernel_size=3):
    input1 = keras.layers.Input(shape=input_shape)
    conv_1 = Convolution2D(8, (kernel_size, kernel_size), padding='same', activation='relu')(input1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    dropout_1 = Dropout(.2)(pool_1)
    conv_2 = Convolution2D(16, (kernel_size, kernel_size), padding='same', activation='relu')(dropout_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropout_2 = Dropout(.2)(pool_2)
    flat_1 = Flatten()(dropout_2)
    dense_1 = Dense(32, activation='relu')(flat_1)
    output = Dense(2)(dense_1)
    return Model(input1, output)



if __name__ == '__main__':
    X, Y = prep_data()
    X_train, X_val = np.split(X, [int(X.shape[0]*.8)], axis=0)
    Y_train, Y_val = np.split(Y, [int(Y.shape[0] * .8)], axis=0)
    model = create_model((X.shape[1], X.shape[2], X.shape[3]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # seed = 7
    # np.random.seed(seed)
    # # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=10, verbose=1)
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    model.fit(X_train,Y_train, validation_data=(X_val,Y_val), verbose=1, epochs=200, batch_size=3)
    a = time.clock()
    model.predict(X)
    print(time.clock()- a)