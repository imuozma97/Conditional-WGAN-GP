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
from architectures.discriminators import Discriminator_concat
from training import Training
from psd_utils import lambda_psd_schedule



trained_models_folder = "Training3D/10-models"
generated_images_folder = "Training3D/10-images"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar, buffer_size = 918)

#Cargamos los datos: número de partículas y redshift
n_part, red = datos.load_npart("Data3D-64.hdf5")
rot1 = datos.rotation(n_part, k=1)
rot2 = datos.rotation(n_part, k=2)
rot3 = datos.rotation(n_part, k=3)

total_data = np.concatenate([n_part, rot1, rot2, rot3], axis = 0)
#Aplicacmos la transformación al número de partículas. Esto es lo que recibirá la red. Ya está en el rango correcto
n_part_transf = datos.transform_npart(n_part, k=11000)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)
#z_vals = np.tile(z_vals, (4, 1))


psd_max, psd_min, mean_psd, psd_sigma, _ = datos.load_psd("PSD_k11000.npz")
#psd_max = np.tile(psd_max, (4, 1))
#psd_min = np.tile(psd_min, (4, 1))
#mean_psd = np.tile(mean_psd, (4, 1))
#psd_sigma = np.tile(psd_sigma, (4, 1))

dataset = datos.crea_dataset(n_part_transf, z_vals, psd_max, psd_min, mean_psd, psd_sigma)

#Cargamos el Discriminador y Generador
generator = Generator_film3(filter1 = 256, filter2 = 128, filter3 = 64)
discriminator = Discriminator_concat(filter1 = 32, filter2 = 64, filter3 = 128, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic1, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = lambda_psd_schedule,
                lambda_term = 10, use_psd = False, use_psd_loss = False)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
