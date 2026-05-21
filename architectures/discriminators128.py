"""
Discriminadores para resolución de 128
"""

import tensorflow as tf
import tensorflow_addons as tfa
from config import embedding_dim

class Discriminator_projection(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3, filter4, layer):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.filter4 = filter4
        self.layer = layer

        if self.layer == "GAP":
            final_layer = tf.keras.layers.GlobalAveragePooling3D()
        if self.layer == "GMP":
            final_layer = tf.keras.layers.GlobalMaxPooling3D()
        if self.layer == "F":
            final_layer = tf.keras.layers.Flatten()
           

        # Embedding del redshift
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
            tf.keras.layers.Dense(embedding_dim, activation='linear'),
        ])

        # Red convolucional modificada
        self.extract_features = tf.keras.Sequential([
            tf.keras.layers.Conv3D(self.filter1, kernel_size=4, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.Conv3D(self.filter2, kernel_size=4, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.Conv3D(self.filter3, kernel_size=3, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.Conv3D(self.filter4, kernel_size=3, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            final_layer
           
        ])
        self.features_dense = tf.keras.layers.Dense(embedding_dim)
        self.final_dense = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02))  # WGAN critic output
        

    def call(self, inputs, training=True, use_psd=False):
        image, z = inputs
        
        f = self.extract_features(image)
        f = self.features_dense(f)
        u = self.final_dense(f)

        z_embed = self.z_embedding(z)

        projection = tf.reduce_sum(f * z_embed, axis = -1, keepdims = True)
        out = u + projection
        return out
