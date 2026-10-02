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
import glob

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
from generate import Fake_images
from config import batch_size1, ncritic3, n_bar2, image_size2, num_cv, num_classes
from architectures.generators128 import Generator_film_linear_swish
from architectures.discriminators128 import Discriminator_projection_swish
from training128 import Training128
from transforms import forward_128
from psd_utils import lambda_psd_schedule3


trained_models_folder = "Training128/2-models"
generated_images_folder = "Training128/2-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar2, buffer_size = 918)
imagenes = Fake_images(N = num_cv, image_size = image_size2, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

#Cargamos los datos: número de partículas y redshift
print("Cargamos los datos")
file = "Training128/datos_reales_final.npz"

with np.load(file) as f:
    forw_final = f["data"]
    z_vals_final = f["labels"]



print("Cargamos psd")
psd_max, psd_min, mean_psd, _, _ = datos.load_psd("PSD_delta-128.npz")

print("Creo dataset")
dataset = datos.crea_dataset(forw_final, z_vals_final, psd_max, psd_min, mean_psd)

#Cargamos el Discriminador y Generador
generator = Generator_film_linear_swish(filter1 = 128, filter2 = 64, filter3 = 32, filter4 = 16)
discriminator = Discriminator_projection_swish(filter1 = 16, filter2 = 32, filter3 = 64, filter4 = 128, layer = "F")


#Cargamos la red principal
print("Llamo a training")
cgan = Training128(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = lambda_psd_schedule3,
                lambda_term = 20, image_size = image_size2, use_psd = False, use_psd_loss = True)

cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))

print("Lamo a train")
cgan.train(dataset, epochs = 20000)
