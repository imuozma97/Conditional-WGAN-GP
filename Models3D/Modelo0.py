"""
Nuevo archivo en el que cambio los datos. Ahora los calculo como arcsinh(alpha*delta) y luego normalizo entre -1 y 1. Alpha = 1
Uso GP de momento, sin SN
Pongo LeakyReLU(0.2) en el Generador_film; además pongo use_bias  = True si no hay BN
El discriminador es el de projection
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
from config import batch_size1, ncritic1, n_bar
from architectures.generators import Generator_film3
from architectures.discriminators import Discriminator_projection
from training import Training
from psd_utils import lambda_psd_schedule



trained_models_folder = "Training3D/0-models"
generated_images_folder = "Training3D/0-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar)


norm_data, z_vals, max_val, min_val= datos.load_data("Data3D-64.hdf5", "global_norm_tanh")
psd_max, psd_min, mean_psd, psd_sigma, _ = datos.load_psd("PSD_norm_c100.npz")
dataset = datos.crea_dataset(norm_data, z_vals, psd_max, psd_min, mean_psd, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film3(filter1 = 256, filter2 = 128, filter3 = 64)
discriminator = Discriminator_projection(filter1 = 32, filter2 = 64, filter3 = 128, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic1, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = lambda_psd_schedule,
                lambda_term = 10, use_psd = False, use_psd_loss = False)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
