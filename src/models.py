import tensorflow as tf
import keras
from tensorflow.keras import layers, regularizers, Model

def build_undercomplete_ae(latent_dim=32, input_dim=784, learning_rate=0.001):
    """
    Builds the Undercomplete Autoencoder.
    Architecture: 784 -> 256 -> 128 -> latent_dim (32) -> 128 -> 256 -> 784
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    x = layers.Dense(256, activation='relu', name='enc_dense_1')(inputs)
    x = layers.Dense(128, activation='relu', name='enc_dense_2')(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    x = layers.Dense(128, activation='relu', name='dec_dense_1')(latent)
    x = layers.Dense(256, activation='relu', name='dec_dense_2')(x)
    outputs = layers.Dense(input_dim, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Undercomplete_AE')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

def build_sparse_ae(latent_dim=32, input_dim=784, l1_reg=1e-5, learning_rate=0.001):
    """
    Builds the Sparse Autoencoder with L1 activity regularization on the latent layer.
    Architecture: 784 -> 256 -> 128 -> latent_dim (32 with L1) -> 128 -> 256 -> 784
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    x = layers.Dense(256, activation='relu', name='enc_dense_1')(inputs)
    x = layers.Dense(128, activation='relu', name='enc_dense_2')(x)
    latent = layers.Dense(
        latent_dim,
        activation='relu',
        activity_regularizer=regularizers.l1(l1_reg),
        name='latent'
    )(x)
    
    x = layers.Dense(128, activation='relu', name='dec_dense_1')(latent)
    x = layers.Dense(256, activation='relu', name='dec_dense_2')(x)
    outputs = layers.Dense(input_dim, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Sparse_AE')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

def build_denoising_ae(latent_dim=64, input_dim=784, learning_rate=0.001):
    """
    Builds the Denoising Autoencoder with higher latent capacity (64 dimensions).
    Architecture: 784 -> 256 -> 128 -> latent_dim (64) -> 128 -> 256 -> 784
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    x = layers.Dense(256, activation='relu', name='enc_dense_1')(inputs)
    x = layers.Dense(128, activation='relu', name='enc_dense_2')(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    x = layers.Dense(128, activation='relu', name='dec_dense_1')(latent)
    x = layers.Dense(256, activation='relu', name='dec_dense_2')(x)
    outputs = layers.Dense(input_dim, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Denoising_AE')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

def build_conv_ae(learning_rate=0.001):
    """
    Builds the Convolutional Autoencoder (74,497 parameters).
    Trained with Binary Cross-Entropy (BCE) to prevent model collapse.
    """
    inputs = layers.Input(shape=(28, 28, 1), name='input')
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv_1')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same', name='enc_pool_1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
    x = layers.MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
    latent = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='latent_map')(x)
    
    # Decoder
    x = layers.UpSampling2D((2, 2), name='dec_upsample_1')(latent)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv_1')(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Convolutional_AE')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy'
    )
    return model

@keras.saving.register_keras_serializable(package="src.models", name="ContractiveAutoencoder")
class ContractiveAutoencoder(Model):
    """
    Custom Contractive Autoencoder:
    Architecture: 784 -> 256 -> 32 (sigmoid) -> 128 -> 784
    Loss = MSE(y_true, y_pred) + lambda_reg * mean( (h * (1 - h))^2 )
    """
    def __init__(self, latent_dim=32, lambda_reg=1e-4, name='Contractive_AE', **kwargs):
        super(ContractiveAutoencoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        
        self.enc_dense_1 = layers.Dense(256, activation='relu', name='enc_dense_1')
        self.latent_layer = layers.Dense(latent_dim, activation='sigmoid', name='latent')
        self.dec_dense_1 = layers.Dense(128, activation='relu', name='dec_dense_1')
        self.output_layer = layers.Dense(784, activation='sigmoid', name='output')
        
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.contractive_loss_tracker = tf.keras.metrics.Mean(name='contractive_loss')

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'lambda_reg': self.lambda_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = self.enc_dense_1(inputs)
        h = self.latent_layer(x)
        x = self.dec_dense_1(h)
        reconstruction = self.output_layer(x)
        return reconstruction

    def get_latent(self, inputs):
        x = self.enc_dense_1(inputs)
        return self.latent_layer(x)

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_loss_tracker, self.contractive_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            x_noisy, x_clean = data
        else:
            x_noisy = x_clean = data

        with tf.GradientTape() as tape:
            x = self.enc_dense_1(x_noisy)
            h = self.latent_layer(x)
            x_rec = self.dec_dense_1(h)
            reconstruction = self.output_layer(x_rec)
            
            recon_loss = tf.reduce_mean(tf.square(x_clean - reconstruction))
            contractive_term = tf.reduce_mean(tf.square(h * (1.0 - h)))
            total_loss = recon_loss + self.lambda_reg * contractive_term

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.contractive_loss_tracker.update_state(contractive_term)
        return {
            'loss': self.loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'contractive_loss': self.contractive_loss_tracker.result()
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x_noisy, x_clean = data
        else:
            x_noisy = x_clean = data

        x = self.enc_dense_1(x_noisy)
        h = self.latent_layer(x)
        x_rec = self.dec_dense_1(h)
        reconstruction = self.output_layer(x_rec)

        recon_loss = tf.reduce_mean(tf.square(x_clean - reconstruction))
        contractive_term = tf.reduce_mean(tf.square(h * (1.0 - h)))
        total_loss = recon_loss + self.lambda_reg * contractive_term

        self.loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.contractive_loss_tracker.update_state(contractive_term)
        return {
            'loss': self.loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'contractive_loss': self.contractive_loss_tracker.result()
        }

def build_contractive_ae(latent_dim=32, lambda_reg=1e-4, learning_rate=0.001):
    model = ContractiveAutoencoder(latent_dim=latent_dim, lambda_reg=lambda_reg)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model
