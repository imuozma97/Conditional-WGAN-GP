"""
Codigo para los discriminadores en 2d
"""

import tensorflow as tf
from config import embedding_dim    

class Discriminator_concat(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        # Embedding del redshift
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
        ])

        # Red convolucional modificada
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filter1, kernel_size = 4, strides=2, padding="same", kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias = True),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv2D(self.filter2, kernel_size=4, strides=2, padding="same",  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias = True),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv2D(self.filter3, kernel_size=4, strides=2, padding="same",kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias = True),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)) 
        ])

    def call(self, inputs, training=True):
        image, z = inputs  # image: (batch, 64, 64, 1), z: (batch, 1)

        z_embed = self.z_embedding(z)  # (batch, embedding_dim)
        z_embed = tf.expand_dims(tf.expand_dims(z_embed, 1), 1)  # (batch, 1, 1, embedding_dim)
        z_embed_broadcast = tf.tile(z_embed, [1, 64, 64, 1])  # (batch, 64, 64, embedding_dim)

        x = tf.concat([image, z_embed_broadcast], axis=-1)  # (batch, 64, 64, 1 + embedding_dim)
        return self.conv_layers(x)