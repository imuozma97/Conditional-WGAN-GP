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
from generate import Fake_images
from config import ncritic3, n_bar_128, image_size_128, num_cv, num_classes
from architectures.generators128 import Generator_film_linear_swish
from architectures.discriminators128 import Discriminator_projection_swish
from training128_dyn import Training128
from transforms import forward_2, backward_2
from psd_utils import lambda_psd_schedule3


trained_models_folder = "Entrenamiento128/0-models"
generated_images_folder = "Entrenamiento128/0-images"

"""
#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar2, buffer_size = 918)

#Cargamos los datos: número de partículas y redshift
n_part, red = datos.load_npart("Data3D-128.hdf5")
delta = datos.delta(n_part)
forw = forward_2(delta+1)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)
"""


#Cargamos las clases necesarias
batch_size1 = 16
datos= Dataset(batch_size1, buffer_size = 918)
imagenes = Fake_images(N = num_cv, image_size = image_size_128, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

#Cargamos los datos: número de partículas y redshift
print("Cargamos los datos")
file = "Entrenamiento128/datos_reales_final_c300.npz"

with np.load(file) as f:
    forw_final = f["data"]
    z_vals_final = f["labels"]


psd_max, psd_min, mean_psd, psd_sigma, _ = datos.load_psd("psd-data/PSD_delta_claude_128.npz")

dataset = datos.crea_dataset(forw_final, z_vals_final, psd_max, psd_min, mean_psd)

#Cargamos el Discriminador y Generador
generator = Generator_film_linear_swish(filter1 = 128, filter2 = 64, filter3 = 32, filter4 = 16)
discriminator = Discriminator_projection_swish(filter1 = 16, filter2 = 32, filter3 = 64, filter4 = 128, layer = "F")


#Cargamos la red principal
cgan = Training128(data_class = datos, discriminator = discriminator, generator = generator,  ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = None,
                lambda_term = 40, image_size = image_size_128, use_psd_loss = False, grad_ratio = 0.5, warmup_epochs = 0)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00002, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
