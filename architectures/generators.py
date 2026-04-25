"""
Este archivo contiene los diferentes modelos de generador que puedo ir utilizando
-Generator_film: Utiliza capas film para introducir la condición z. Empieza con un cubo 8x8x8x256, que en Artemisa solo puedo
                 utilizarlo con batch_size1 = 17
                 
-Generator_film2: Utiliza capas film para introducir la condición z. Empieza con un cubo 4x4x4x256, que en Artemisa solo puedo
                 utilizarlo con batch_size1 = 34
"""
import tensorflow as tf
import os
import numpy as np

from config import latent_dim, num_classes, N, image_size, boxsize, mass,embedding_dim

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
        

class Generator_film(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dense(embedding_dim),
        ])

       #Cubo inicial y reshape
        self.dense = tf.keras.layers.Dense(8 * 8 * 8 * filter1)
        self.reshape = tf.keras.layers.Reshape((8, 8, 8, filter1))

        # Bloque 1
        self.conv1 = tf.keras.layers.Conv3DTranspose(filter1, 4, strides=2, padding='same', use_bias=False)
        self.film1 = FiLMLayer(filter1)
        self.act1 = tf.keras.layers.ReLU()

        # Bloque 2
        self.conv2 = tf.keras.layers.Conv3DTranspose(filter2, 4, strides=2, padding='same', use_bias=False)
        self.film2 = FiLMLayer(filter)
        self.act2 = tf.keras.layers.ReLU()

        # Bloque 3
        self.conv3 = tf.keras.layers.Conv3DTranspose(filter3, 3, strides=2, padding='same', use_bias=False)
        self.film3 = FiLMLayer(filter4)
        self.act3 = tf.keras.layers.ReLU()

        #Última capa
        self.out = tf.keras.layers.Conv3D(1, 3, padding='same', activation='tanh')

    def call(self, inputs, training=True):
        z_latent, z_condition = inputs

        # Condición por el embedding
        z = self.z_embedding(z_condition)

        # Dense y reshape del ruido---¿Debería poner también la condición desde el principio?
        x = self.dense(z_latent)
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

        return self.out(x)



class Generator_film2(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3, filter4):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dense(embedding_dim),
        ])

        # Base
        self.dense = tf.keras.layers.Dense(4 * 4 * 4 * filter1)
        self.reshape = tf.keras.layers.Reshape((4, 4, 4, filter1))

        # Bloque 1
        self.conv1 = tf.keras.layers.Conv3DTranspose(filter1, 4, strides=2, padding='same', use_bias=False)
        self.film1 = FiLMLayer(filter1)
        self.act1 = tf.keras.layers.ReLU()

        # Bloque 2
        self.conv2 = tf.keras.layers.Conv3DTranspose(filter2, 4, strides=2, padding='same', use_bias=False)
        self.film2 = FiLMLayer(filter2)
        self.act2 = tf.keras.layers.ReLU()

        # Bloque 3
        self.conv3 = tf.keras.layers.Conv3DTranspose(filter3, 4, strides=2, padding='same', use_bias=False)
        self.film3 = FiLMLayer(filter3)
        self.act3 = tf.keras.layers.ReLU()

        # Bloque 4
        self.conv4 = tf.keras.layers.Conv3DTranspose(filter4, 3, strides=2, padding='same', use_bias=False)
        self.film4 = FiLMLayer(filter4)
        self.act4 = tf.keras.layers.ReLU()

        # Output
        self.out = tf.keras.layers.Conv3D(1, 3, padding='same', activation='tanh')

    def call(self, inputs, training=True):
        z_latent, z_condition = inputs

        # Embedding de condición
        z = self.z_embedding(z_condition)

        # Base SOLO con ruido
        x = self.dense(z_latent)
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


        

class Generator_concat(tf.keras.Model): #TAL VEZ TENDRÍA QUE QUITAR BACHNORM
    def __init__(self, filter1, filter2, filter3):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        
        #Embedding de la condición
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
        ])

        self.net = tf.keras.Sequential([
            # Capa densa inicial para crear un cubo base 4x4x4 con 128 canales
            tf.keras.layers.Dense(8 * 8 * 8 * filter1),
            tf.keras.layers.Reshape((8, 8, 8, filter1)), 

            # Bloque1
            tf.keras.layers.Conv3DTranspose(filter1, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # Bloque2
            tf.keras.layers.Conv3DTranspose(filter2, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            
            #Bloque3
            tf.keras.layers.Conv3DTranspose(filter3, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # Capa de salida
            tf.keras.layers.Conv3D(1, kernel_size=3, padding='same', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)])

    def call(self, inputs, training=True):
        z_latent, z_condition = inputs  
        z_embed = self.z_embedding(z_condition)  
        concat_input = tf.concat([z_latent, z_embed], axis=-1)
        return self.net(concat_input)
