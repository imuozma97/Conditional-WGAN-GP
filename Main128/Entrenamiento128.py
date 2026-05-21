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
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False, "constant_folding": True, "shape_optimization": True, "arithmetic_optimization": True, "disable_meta_optimizer": False, "function_optimization": True})


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from preprocess_data import Dataset
from config import batch_size1, ncritic3, n_bar2
from architectures.generators128 import Generator_film_linear
from architectures.discriminators128 import Discriminator_projection
from training import Training
from psd_utils import lambda_psd_schedule



trained_models_folder = "Results3D-128/0-models"
generated_images_folder = "Results3D-128/0-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar2)


norm_data, z_vals, _, _ = datos.load_data("redshift_norm", "Data3D-128.hdf5")
psd_max, psd_min, _, psd_sigma, all_psd = datos.load_psd("PSD_norm_mu_sigma_c100.npz")
dataset = datos.crea_dataset(norm_data, z_vals, psd_max, psd_min, all_psd, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film_linear(filter1 = 256, filter2 = 128, filter3 = 64, 32)
discriminator = Discriminator_projection(filter1 = 32, filter2 = 64, filter3 = 128, , 256, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = lambda_psd_schedule,
                lambda_term = 20, use_psd = False, use_psd_loss = False)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
