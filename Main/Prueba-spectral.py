"""
Archivo principal para entrenar la red. Este simula a Artemisa3, que contiene:
D + projection+Flatten
G + film+linear, sin BN; 8x8x8x256
batch_size = 17
n_critic = 3
latent_dim = 128

Este archivo utiliza datos noramlizados y PSD también normalizado.

COSAS IMPORTANTES:
Utilizo norm. En caso de usar desnorm, sobrarían las salidas de max_desnorm y min_desnorm.
En este caso backward es None también, no se necesita invertir los datos
"""
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_XLA_ENABLE'] = '0'

import tensorflow as tf
import tensorflow.keras as keras


import numpy as np
# Optimizaciones de memoria
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False, "constant_folding": True, "shape_optimization": True, "arithmetic_optimization": True, "disable_meta_optimizer": False, "function_optimization": True})


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from preprocess_data import Dataset
from config import batch_size1, ncritic3
from architectures.generators import Generator_film_linear
from architectures.discriminators import Discriminator_projection, Spectral_Discriminator
from training2 import Training2
from psd_utils import lambda_psd_schedule



trained_models_folder = "Results3D/prueba-models"
generated_images_folder = "Results3D/prueba-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1)


norm_data, z_vals, _, _ = datos.load_data("redshift_norm")
psd_max, psd_min, _, psd_sigma, psd_real= datos.load_psd("PSD_norm_mu_sigma_c100.npz")
dataset = datos.crea_dataset(norm_data, z_vals, psd_max, psd_min, psd_real, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film_linear(filter1 = 256, filter2 = 128, filter3 = 64)
discriminator1 = Discriminator_projection(filter1 = 64, filter2 = 128, filter3 = 256, layer = "GAP")
discriminator2 = Spectral_Discriminator()


#Cargamos la red principal
cgan = Training2(data_class = datos, discriminator1 = discriminator1, discriminator2 = discriminator2, generator = generator, batch_size = batch_size1, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder,
                use_psd = True, use_psd_loss = False)
cgan.compile(d_optimizer1 = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9), 
             d_optimizer2 = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
