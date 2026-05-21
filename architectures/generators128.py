"""
Archivo para Genreadores en 128
"""

import tensorflow as tf
import os
import numpy as np

from config import embedding_dim


class FiLMLayer(tf.keras.layers.Layer):
    def __init__(self, n_channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2 * n_channels)

    def call(self, x, z):
        gamma_beta = self.dense(z)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)

        gamma = gamma[:, None, None, None, :]
        beta  = beta[:, None, None, None, :]

        return gamma * x + beta


class Generator_film_linear(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3, filter4):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.filter4 = filter4

        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dense(embedding_dim),
        ])

       #Cubo inicial y reshape
        self.dense = tf.keras.layers.Dense(8 * 8 * 8 * filter1)
        self.reshape = tf.keras.layers.Reshape((8, 8, 8, filter1))

        # Bloque 1
        self.conv1 = tf.keras.layers.Conv3DTranspose(filter1, 4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias=False)
        self.film1 = FiLMLayer(filter1)
        self.act1 = tf.keras.layers.ReLU()

        # Bloque 2
        self.conv2 = tf.keras.layers.Conv3DTranspose(filter2, 4, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias=False)
        self.film2 = FiLMLayer(filter2)
        self.act2 = tf.keras.layers.ReLU()

        # Bloque 3
        self.conv3 = tf.keras.layers.Conv3DTranspose(filter3, 3, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias=False)
        self.film3 = FiLMLayer(filter3)
        self.act3 = tf.keras.layers.ReLU()


        self.conv4 = tf.keras.layers.Conv3DTranspose(filter4, 3, strides=2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias=False)
        self.film4 = FiLMLayer(filter4)
        self.act4 = tf.keras.layers.ReLU()

        #Última capa
        self.out = tf.keras.layers.Conv3D(1, 3, padding='same', activation='linear')

    def call(self, inputs, training=True):
        z_latent, z_condition = inputs

        # Condición por el embedding
        z = self.z_embedding(z_condition)

        concat = tf.concat([z_latent, z], axis=-1)

        x = self.dense(concat)
        x = self.reshape(x)

        # Bloques con FiLM
        x = self.conv1(x)
        x = self.film1(x, z)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.film2(x, z)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.film3(x, z)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.film4(x, z)
        x = self.act4(x)

        return self.out(x)

