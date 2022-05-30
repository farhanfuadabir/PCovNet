import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def lstm_autoencoder(n_timesteps, n_features, name="LSTM_Autoencoder"):
    inputs = tf.keras.Input(shape=(n_timesteps, n_features))
    x = layers.LSTM(units=128, return_sequences=True, name="encoder1")(inputs)
    x = layers.LSTM(units=64, return_sequences=False, name="encoder2")(x)
    x = layers.RepeatVector(n=n_timesteps, name="repeat_vec")(x)
    x = layers.LSTM(units=64, return_sequences=True, name="decoder1")(x)
    x = layers.LSTM(units=128, return_sequences=True, name="decoder2")(x)
    outputs = layers.TimeDistributed(layers.Dense(units=n_features),
                                     name="reconst")(x)

    return tf.keras.Model(inputs, outputs, name=name)
