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
from config import batch_size1, n_bar_64, n_bar_128, image_size_64, image_size_128
from architectures.generators import Generator_upscaling
from training_upsc import Training_upsc
from transforms import forward_2


trained_models_folder = "Fase2-Training64-128/0-models"
generated_images_folder = "Fase2-Training64-128/0-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, buffer_size = 918)

#Cargamos los datos: número de partículas y redshift
n_part, red = datos.load_npart("Data3D-64.hdf5")  # SEPARAR ENTRE TRAIN Y TEST
delta = datos.delta(n_part, n_bar_64)
#forw = forward_2(delta+1)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)

_, _, _, _, all_psd_128 = datos.load_psd("PSD_delta_128.npz")

dataset = datos.crea_dataset(delta, z_vals, all_psd_128)

#Cargamos el Discriminador y Generador
generator = Generator_upscaling(filter1 = 64)

#Cargamos la red principal
cgan = Training_upsc(generator = generator, batch_size = batch_size1, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, image_size = image_size_128)
    
cgan.compile(g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
