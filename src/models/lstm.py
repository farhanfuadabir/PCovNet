import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def basic_lstm(n_timesteps, n_features, name="Basic_LSTM"):
    inputs = tf.keras.Input(shape=(n_timesteps, n_features))
    x = layers.LSTM(units=64,
                    return_sequences=True,
                    name='lstm1')(inputs)
    x = layers.LSTM(units=64,
                    return_sequences=True,
                    name='lstm2')(x)
    outputs = layers.LSTM(units=n_features,
                          return_sequences=True,
                          activation=None)(x)

    return tf.keras.Model(inputs, outputs, name=name)
