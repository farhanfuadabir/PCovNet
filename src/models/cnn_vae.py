import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Sampling(layers.Layer):
    def __init__(self, name='sampling'):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(
            shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def cnn_encoder(n_timesteps, n_channels, latent_dim, name='Encoder'):
    encoder_input = keras.Input(shape=(n_timesteps, n_channels),
                                name='encoder_input')
    x = layers.Conv1D(filters=128,
                      kernel_size=3,
                      activation='relu',
                      strides=2,
                      padding='same',
                      name='encoder1')(encoder_input)
    x = layers.Conv1D(filters=64,
                      kernel_size=3,
                      activation='relu',
                      strides=2,
                      padding='same',
                      name='encoder2')(x)
    x = layers.Conv1D(filters=32,
                      kernel_size=3,
                      activation='relu',
                      strides=2,
                      padding='same',
                      name='encoder3')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(latent_dim,
                     activation='relu',
                     name='dense')(x)
    z_mean = layers.Dense(units=latent_dim,
                          name='z_mean')(x)
    z_log_var = layers.Dense(units=latent_dim,
                             name='z_var')(x)
    z = Sampling(name='z')([z_mean, z_log_var])

    return keras.Model(encoder_input, [z_mean, z_log_var, z], name=name)


def cnn_decoder(n_timesteps, n_channels, latent_dim, name='Decoder'):
    # (n_timesteps // (2 ** n_conv_layers), filters_last_encoder_cnn)
    shape_init = [n_timesteps // (2**3), 32]

    decoder_input = keras.Input(shape=(latent_dim,),
                                name='decoder_input')
    x = layers.Dense((np.prod(shape_init)),
                     activation='relu',
                     name='dense')(decoder_input)
    x = layers.Reshape(shape_init)(x)
    x = layers.Conv1DTranspose(filters=32,
                               kernel_size=3,
                               activation='relu',
                               strides=2,
                               padding='same',
                               name='decoder1')(x)
    x = layers.Conv1DTranspose(filters=64,
                               kernel_size=3,
                               activation='relu',
                               strides=2,
                               padding='same',
                               name='decoder2')(x)
    x = layers.Conv1DTranspose(filters=128,
                               kernel_size=3,
                               activation='relu',
                               strides=2,
                               padding='same',
                               name='decoder3')(x)
    reconst = layers.Conv1DTranspose(filters=1,
                                     kernel_size=3,
                                     activation='sigmoid',
                                     padding='same')(x)

    return keras.Model(decoder_input, reconst, name='Decoder')


class VAE(keras.Model):
    def __init__(self,
                 n_timesteps,
                 n_channels,
                 latent_dim,
                 name='Variational_AutoEncoder'):
        super(VAE, self).__init__(name=name)
        # Set Instance Variables
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.model_name = name
        # Set Encoder
        self.encoder = cnn_encoder(self.n_timesteps,
                                   self.n_channels,
                                   self.latent_dim)
        # Set Decoder
        self.decoder = cnn_decoder(self.n_timesteps,
                                   self.n_channels,
                                   self.latent_dim)
        # Set Loss Trackers
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconst_loss_tracker = keras.metrics.Mean(name="reconst_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconst_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Record Data during Training Pass
        with tf.GradientTape() as tape:
            # Get Reconstructed Data
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconst_data = self.decoder(z, training=True)
            # Calculate Reconstruction Loss
            reconst_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.mean_squared_error(data, reconst_data), axis=1)
            )
            # Calculate KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # Calculate Total Loss
            total_loss = reconst_loss + kl_loss
        # Calculate Gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Update Gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update Metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconst_loss_tracker.update_state(reconst_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # Return Loss Dictionary
        return {
            "loss": self.total_loss_tracker.result(),
            "reconst_loss": self.reconst_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        # Get Reconstructed Data
        z_mean, z_log_var, z = self.encoder(inputs)
        reconst_inputs = self.decoder(z)
        # Calculate Reconstruction Loss
        reconst_loss = tf.reduce_mean(tf.reduce_sum(
            keras.losses.mean_squared_error(inputs, reconst_inputs), axis=1)
        )
        # Calculate KL Divergence Loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        # Calculate Total Loss
        total_loss = reconst_loss + kl_loss
        # Add Losses
        self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        self.add_metric(total_loss, name='loss', aggregation='mean')
        self.add_metric(reconst_loss, name='reconst_loss', aggregation='mean')
        # Return Reconstructed Data
        return reconst_inputs

    def build_graph(self):
        x = keras.Input(shape=(self.n_timesteps, self.n_channels),
                        name='Input')
        return keras.Model(inputs=[x],
                           outputs=self.call(x),
                           name=self.model_name)

    def encoder_predict(self, inputs):
        _, _, z = self.encoder(inputs)
        return z.numpy()

    def decoder_predict(self, inputs):
        return self.decoder(inputs).numpy()

    def get_pred(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z).numpy()

    def get_loss_array(self, inputs):
        # Get Reconstructed Data
        z_mean, z_log_var, z = self.encoder(inputs)
        reconst_inputs = self.decoder(z)
        # Calculate Reconstruction Loss
        reconst_loss = tf.reduce_sum(
            keras.losses.mean_squared_error(inputs, reconst_inputs), axis=1)
        # Calculate KL Divergence Loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        # Calculate Total Loss
        total_loss = reconst_loss + kl_loss

        return total_loss.numpy()

    def print_summary(self):
        self.encoder.summary()
        print("\n")
        self.decoder.summary()
        print("\n")
        self.build_graph().summary()
