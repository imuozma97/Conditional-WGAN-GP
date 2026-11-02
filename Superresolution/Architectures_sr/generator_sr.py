import tensorflow as tf
from config_sr import embedding_dim
from architectures.generators import FiLMLayer


class Generator_sr(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation = 'swish'),
            tf.keras.layers.Dense(embedding_dim, activation = 'swish'),
        ])
   
        # Bloque 1
        self.conv1 = tf.keras.layers.Conv3DTranspose(filter1, 3, strides = 1, padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias = True)
        self.film1 = FiLMLayer(filter1)
        self.act1 = tf.keras.layers.ReLU()

        # Bloque 2
        self.conv2 = tf.keras.layers.Conv3DTranspose(filter2, 4, strides = 2, padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias = True)
        self.film2 = FiLMLayer(filter2)
        self.act2 = tf.keras.layers.ReLU()

        # Bloque 3
        self.conv3 = tf.keras.layers.Conv3DTranspose(filter3, 3, strides = 1, padding = 'same', kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02),
        use_bias = True)
        self.film3 = FiLMLayer(filter3)
        self.act3 = tf.keras.layers.ReLU()

        #Última capa
        self.out = tf.keras.layers.Conv3D(1, 3, padding = 'same', activation = 'linear',  kernel_initializer = tf.keras.initializers.RandomNormal(0.0, 0.02))

    def call(self, inputs, training=True):
        cube, z_condition = inputs

        # Condición por el embedding
        z = self.z_embedding(z_condition, training = training)


        # Bloques con FiLM
        x = self.conv1(cube)
        x = self.film1(x, z)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.film2(x, z)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.film3(x, z)
        x = self.act3(x)

        return self.out(x, training = training)