import tensorflow as tf
from config_sr import embedding_dim

class Discriminator_sr(tf.keras.Model):
    def __init__(self, filter1, filter2, filter3, layer):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3

        self.layer = layer

        # Amplía el cubo LR al tamaño del HR (vecino más cercano)
        self.up_lr = tf.keras.layers.UpSampling3D(2)

        if self.layer == "GAP":
            final_layer = tf.keras.layers.GlobalAveragePooling3D()
        if self.layer == "GMP":
            final_layer = tf.keras.layers.GlobalMaxPooling3D()
        if self.layer == "F":
            final_layer = tf.keras.layers.Flatten()
           

        # Embedding del redshift
        self.z_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='swish'),
            tf.keras.layers.Dense(embedding_dim, activation='swish'),
        ])

        # Capas convolucionales que estraen las características de la imagen y pasan por la capa final Flaten para aplanarlas
        self.extract_features = tf.keras.Sequential([
            tf.keras.layers.Conv3D(self.filter1, kernel_size=4, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv3D(self.filter2, kernel_size=4, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv3D(self.filter3, kernel_size=4, strides=2, padding="same",
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            final_layer
           
        ])
        
        #Capa final dense, que saca el score de la wgan (hasta aquí sería una wgan normal sin condicionar)
        self.final_dense = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02))  # WGAN critic output
        
        #Esta capa es para embeber las caracteristicas de la imagen y que estén en el mismo espacio que las condiciones
        self.features_dense = tf.keras.layers.Dense(embedding_dim)
        

    def call(self, inputs, training=True):
        cube_hr, cube_lr, z = inputs

        #Concatenamos los dos cubos, ampliando el tamaño del de LR
        x = tf.concat([cube_hr, self.up_lr(cube_lr)], axis=-1)

        # Primero, sacamos el mapa de características 3D aplanado de la imagen
        f = self.extract_features(x, training = training)

        #Después proyectamos las características al espacio del embedding
        f_projected = self.features_dense(f, training = training)

        #Sacamos el score de la wgan
        u = self.final_dense(f, training = training)

        #Embebemos la condición
        z_embed = self.z_embedding(z, training = training)

        #Producto interno de los embebidos
        projection = tf.reduce_sum(f_projected * z_embed, axis = -1, keepdims = True)

        out = u + projection

        return out
