import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from numpy import matlib as nm
import pandas as pd
from sklearn import preprocessing
from keras import utils

def disease_auto_encoder(y_train):
    encoding_dim = 64
    input_vector = layers.Input(shape=(432,))

    # encoder layer
    encoded = layers.Dense(350, activation='relu')(input_vector)
    encoded = layers.Dense(250, activation='relu')(encoded)
    encoded = layers.Dense(150, activation='relu')(encoded)
    encoded = layers.Dense(100, activation='relu')(encoded)
    disease_encoder_output = layers.Dense(encoding_dim)(encoded)

    # decoder layer
    decoded = layers.Dense(100, activation='relu')(disease_encoder_output)
    decoded = layers.Dense(150, activation='relu')(decoded)
    decoded = layers.Dense(250, activation='relu')(decoded)
    decoded = layers.Dense(350, activation='relu')(decoded)
    decoded = layers.Dense(432, activation='tanh')(decoded)

    # build a autoencoder model
    autoencoder = models.Model(inputs=input_vector, outputs=decoded)
    encoder = models.Model(inputs=input_vector, outputs=disease_encoder_output)

    # activate model
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(y_train, y_train, epochs=20, batch_size=100, shuffle=True)
    disease_encoded_vector = encoder.predict(y_train)
    return disease_encoded_vector


def five_AE(d_sim):

    dtrain, label = data_process(d_sim)
    d_features = disease_auto_encoder(dtrain)
    return d_features

def data_process(d_sim):


    A = pd.read_csv("mydata/data/M_D.csv", index_col=0).to_numpy()
    R_A = np.repeat(A, repeats=216, axis=0)
    sd = nm.repmat(d_sim, 2262, 1)
    train1 = np.concatenate((R_A, sd), axis=1)
    label = A.reshape((488592, 1))

    return train1, label






