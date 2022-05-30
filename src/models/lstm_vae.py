import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def __init__(self, name='sampling'):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(
            shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# class Encoder(layers.Layer):
# 	def __init__(self, latent_dim, name='Encoder'):
# 		super(Encoder, self).__init__(name=name)
# 		self.model_name = name
# 		self.encoder_1 = layers.LSTM(units=128,
# 									return_sequences=True,
# 									name='encoder_1')
# 		self.encoder_2 = layers.LSTM(units=64,
# 									return_sequences=False,
# 									name='encoder_2')
# 		self.dense_mean = layers.Dense(units=latent_dim,
# 									name='dense_mean')
# 		self.dense_var = layers.Dense(units=latent_dim,
# 									name='dense_var')
# 		self.sampling = Sampling()

# 	def call(self, input_tensor, training=False):
# 		x = self.encoder_1(input_tensor, training=training)
# 		x = self.encoder_2(x, training=training)
# 		z_mean = self.dense_mean(x, training=training)
# 		z_log_var = self.dense_var(x, training=training)
# 		z = self.sampling(inputs=[z_mean, z_log_var])
# 		return z_mean, z_log_var, z

# 	def build_graph(self):
# 		x = keras.Input(shape=(self.n_timesteps, self.n_channels),
#                  		name='Input')
# 		return keras.Model(inputs=[x],
# 							outputs=self.call(x),
#                      		name=self.model_name)


# class Decoder(layers.Layer):
# 	def __init__(self, n_timesteps, n_channels, name='Decoder'):
# 		super(Decoder, self).__init__(name=name)
# 		self.model_name = name
# 		self.embeddings = layers.RepeatVector(n=n_timesteps,
# 											name='embeddings')
# 		self.decoder_1 = layers.LSTM(units=64,
# 									return_sequences=True,
# 									name='decoder_1')
# 		self.decoder_2 = layers.LSTM(units=128,
# 									return_sequences=True,
# 									name='decoder_2')
# 		self.time_dist_dense = layers.TimeDistributed(
# 			layers.Dense(units=n_channels),
# 			name='dense_t_dist')

# 	def call(self, input_tensor, training=False):
# 		z = self.embeddings(input_tensor, training=training)
# 		x = self.decoder_1(z, training=training)
# 		x = self.decoder_2(x, training=training)
# 		x = self.time_dist_dense(x, training=training)
# 		return x

    # def build_graph(self):
    # 	x = keras.Input(shape=(self.n_timesteps, self.n_channels),
    #              		name='Input')
    # 	return keras.Model(inputs=[x],
    # 						outputs=self.call(x),
    #                  		name=self.model_name)


def lstm_encoder(n_timesteps, n_channels, latent_dim, name='Encoder'):
    encoder_input = keras.Input(
        shape=(n_timesteps, n_channels), name='encoder_input')
    x = layers.LSTM(units=128,
                    return_sequences=True,
                    name='encoder_1',)(encoder_input)
    x = layers.LSTM(units=64,
                    return_sequences=False,
                    name='encoder_2')(x)
    z_mean = layers.Dense(units=latent_dim,
                          name='z_mean')(x)
    z_log_var = layers.Dense(units=latent_dim,
                             name='z_var')(x)
    z = Sampling(name='z')([z_mean, z_log_var])

    return keras.Model(encoder_input, [z_mean, z_log_var, z], name=name)


def lstm_decoder(n_timesteps, n_channels, latent_dim, name='Decoder'):
    decoder_input = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.RepeatVector(n=n_timesteps,
                            name='embeddings')(decoder_input)
    x = layers.LSTM(units=64,
                    return_sequences=True,
                    name='decoder_1')(x)
    x = layers.LSTM(units=128,
                    return_sequences=True,
                    name='decoder_2')(x)
    reconst = layers.TimeDistributed(layers.Dense(units=n_channels),
                                     name='dense')(x)

    return keras.Model(decoder_input, reconst, name=name)


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
        self.encoder = lstm_encoder(self.n_timesteps,
                                    self.n_channels,
                                    self.latent_dim)
        # Set Decoder
        self.decoder = lstm_decoder(self.n_timesteps,
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
        return self.encoder(inputs)

    def decoder_predict(self, inputs):
        return self.decoder(inputs)

    def get_pred(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

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
