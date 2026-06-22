"""
Datos3 + Arquitectura1
Datos: forward(delta+1)+salida 'Linear' en el Generador
Generator_film_linear
Discriminator_projection
"""
import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_XLA_ENABLE'] = '0'

# Optimizaciones de memoria
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False, "constant_folding": True, "shape_optimization": True, "arithmetic_optimization": True, "disable_meta_optimizer": False, "function_optimization": True})
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from preprocess_data import Dataset
from config import batch_size2, ncritic3, n_bar
from architectures.generators import Generator_film_linear_swish
from architectures.discriminators import Discriminator_projection_swish
from training import Training
from transforms import forward_2
from psd_utils import lambda_psd_schedule


trained_models_folder = "Training3D/13-models"
generated_images_folder = "Training3D/13-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size2, n_bar, buffer_size = 918)

#Cargamos los datos: número de partículas y redshift
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part)
forw = forward_2(delta+1)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)


psd_max, psd_min, mean_psd, psd_sigma, _ = datos.load_psd("PSD_delta.npz")

dataset = datos.crea_dataset(forw, z_vals, psd_max, psd_min, mean_psd, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film_linear_swish(filter1 = 256, filter2 = 128, filter3 = 64)
discriminator = Discriminator_projection_swish(filter1 = 32, filter2 = 64, filter3 = 128, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size2, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = lambda_psd_schedule,
                lambda_term = 20, use_psd = False, use_psd_loss = False)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
