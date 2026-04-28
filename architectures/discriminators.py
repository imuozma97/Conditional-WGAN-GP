"""
Archivo que contiene los diferentes tipos de Discriminadores que puedo llegar a utilizar:

- Discriminator_projection: Utiliza la proyección de Miyato para introducir las condiciones. 
- Discriminator_concat: Este concatena las condiciones a las imágenes. En artemisa no cabe
- Discriminador_psd: Utiliza también la proyección, pero antes introduce el psd en otra rama, usando el logairitmo
"""

import tensorflow as tf
from config import embedding_dim

class Discriminator_projection(tf.keras.Model):
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

            tf.keras.layers.GlobalAveragePooling3D(),
           
        ])
        self.features_dense = tf.keras.layers.Dense(embedding_dim)
        self.final_dense = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02))  # WGAN critic output
        

    def call(self, inputs, training=True):
        image, z = inputs
        
        f = self.extract_features(image)
        f = self.features_dense(f)
        u = self.final_dense(f)

        z_embed = self.z_embedding(z)

        projection = tf.reduce_sum(f * z_embed, axis = -1, keepdims = True)
        out = u + projection
        return out



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
            tf.keras.layers.Conv3D(self.filter1, kernel_size=4, strides=1, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.Conv3D(self.filter2, kernel_size=4, strides=1, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.Conv3D(self.filter3, kernel_size=3, strides=1, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
           # tf.keras.layers.MaxPooling3D(pool_size=2),

            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02))  # WGAN critic output
        ])

    def call(self, inputs, training=True):
        image, z = inputs  

        z_embed = self.z_embedding(z) 
        z_embed = tf.expand_dims(tf.expand_dims(tf.expand_dims(z_embed, 1), 1), 1)
        z_embed_broadcast = tf.tile(z_embed, [1, 64, 64, 64, 1]) 

        x = tf.concat([image, z_embed_broadcast], axis=-1)  
        return self.conv_layers(x)




class Discriminator_psd(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        # embedding de z (para projection)
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dense(embedding_dim),
        ])

        # rama espacial
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv3D(filter1, 4, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
           # tf.keras.layers.MaxPooling3D(2),

            tf.keras.layers.Conv3D(filter2, 4, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
           # tf.keras.layers.MaxPooling3D(2),

            tf.keras.layers.Conv3D(filter3, 3, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),
            #tf.keras.layers.MaxPooling3D(2),

            tf.keras.layers.GlobalAveragePooling3D(),
        ])

        # rama PSD
        self.psd_branch = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(embedding_dim),
        ])

        # capas finales
        self.features_dense = tf.keras.layers.Dense(embedding_dim)
        self.final_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=True, use_psd=True):
        
        if use_psd: 
            image, z, psd = inputs
        else:
            image, z = inputs

        f_img = self.conv_layers(image)
        f_img = self.features_dense(f_img)

        if use_psd:
            f_psd = self.psd_branch(psd)
            f = f_img + f_psd
        else:
            f = f_img  

        z_embed = self.z_embedding(z)
        projection = tf.reduce_sum(f * z_embed, axis=-1, keepdims=True)

        out = self.final_dense(f)

        return out + projection
