"""
Generadores en 2d
"""

import tensorflow as tf
from config import embedding_dim, latent_dim



class Generator_concat(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        # Red para mapear z (float32, 1 dim) a embedding vector
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
        ])

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(8 * 8 * self.filter1),
            tf.keras.layers.Reshape((8, 8, self.filter1)),

            # Upsampling 1: 8x8 -> 16x16
            #tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(self.filter1, kernel_size=5, strides = 2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02), use_bias = True),
            tf.keras.layers.LeakyReLU(0.2),

            # Upsampling 2: 16x16 -> 32x32
            #tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(self.filter2, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02), use_bias = True),
            tf.keras.layers.LeakyReLU(0.2),

            # Upsampling 3: 32x32 -> 64x64
            #tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(self.filter3, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            # Salida final
            tf.keras.layers.Conv2D(1, kernel_size=3, padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                                   activation='tanh')
        ])

        
    def call(self, inputs, training=True):
        z_latent, z_condition = inputs  
        z_embed = self.z_embedding(z_condition)  
        concat_input = tf.concat([z_latent, z_embed], axis=1)
        return self.net(concat_input)





class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Red para mapear z (float32, 1 dim) a embedding vector
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
        ])

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(8 * 8 * 128),
            tf.keras.layers.Reshape((8, 8, 128)),

            # Upsampling 1: 8x8 -> 16x16
            tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=5, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # Upsampling 2: 16x16 -> 32x32
            tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(256, kernel_size=4, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # Upsampling 3: 32x32 -> 64x64
            tf.keras.layers.UpSampling2D(interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(512, kernel_size=4, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02), use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # Salida final
            tf.keras.layers.Conv2D(1, kernel_size=3, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   activation='tanh')
        ])

        
    def call(self, inputs, training=True):
        z_latent, z_condition = inputs  # z_latent: (batch, latent_dim), z_condition: (batch, 1)
        z_embed = self.z_embedding(z_condition)  # -> (batch, embedding_dim)
        concat_input = tf.concat([z_latent, z_embed], axis=1)
        return self.net(concat_input)