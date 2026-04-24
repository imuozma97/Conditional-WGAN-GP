"""
Archivo principal para entrenar la red. Este simula a Artemisa3, que contiene:
D + psd
G + film, sin BN; 8x8x8x256
batch_size = 17
n_critic = 2
latent_dim = 128

Este archivo utiliza datos noramlizados y PSD también normalizado.

COSAS IMPORTANTES:
Utilizo norm. En caso de usar desnorm, sobrarían las salidas de max_desnorm y min_desnorm.
En este caso backward es None también, no se necesita invertir los datos
"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from preprocess_data import Dataset
from transforms import backward
from config import embedding_dim, batch_size1, ncritic2
from models.generators import Generator_film
from models.discriminators import Discriminator_psd
from power import Power
from training import Training



trained_models_folder = "Results3D/0-models"
generated_images_folder = "Results3D/0-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1)
power = Power()


norm_data, z_vals, _, _ = datos.load_data("norm")
psd_max, psd_min, psd_mean, psd_sigma = datos.load_psd("norm")
dataset = datos.crea_dataset(norm_data, z_vals, psd_max, psd_min, psd_mean, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film(embedding_dim = embedding_dim, filter1 = 256, filter2 = 128, filter3 = 64)
discriminator = Discriminator_psd(embedding_dim=embedding_dim, filter1 = 32, filter2 = 64, filter3 = 128)


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic2, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder)

cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
