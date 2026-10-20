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
from config import batch_size1, ncritic3, n_bar_64, image_size_64
from architectures.generators import Generator_film_linear_swish
from architectures.discriminators import Discriminator_projection_swish
from training_dyn import Training
from transforms import forward_2
from psd_utils import lambda_psd_schedule


trained_models_folder = "Training3D/17-models_dyn"
generated_images_folder = "Training3D/17-images_dyn"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, buffer_size = 918)

#Cargamos los datos: número de partículas y redshift
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64)
forw = forward_2(delta+1)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)


# Índices a eliminar en cada bloque
eliminar = [4, 12, 25]
mask = np.ones(34, dtype=bool)
mask[eliminar] = False

# Reorganizar en bloques de 34
z_vals_reshape = z_vals.reshape(-1, 34)
z_vals_final = z_vals_reshape[:, mask].reshape(-1, 1)

forw = forw.numpy()
cubos_final = forw.reshape(-1, 34, 64, 64, 64, 1)[:, mask]
cubos_final = cubos_final.reshape(-1, 64, 64, 64, 1)


psd_max, psd_min, mean_psd, psd_sigma, _ = datos.load_psd("psd-data/PSD_delta_claude.npz")

psd_max_reshape = psd_max.reshape(-1, 34, 31)
psd_max_final = psd_max_reshape[:, mask].reshape(-1, 31)

psd_min_reshape = psd_min.reshape(-1, 34, 31)
psd_min_final = psd_min_reshape[:, mask].reshape(-1, 31)

mean_psd_reshape = mean_psd.reshape(-1, 34, 31)
mean_psd_final = mean_psd_reshape[:, mask].reshape(-1, 31)



#837 cubos no es múltiplo de 17: se descarta el último batch incompleto de cada época
dataset = tf.data.Dataset.from_tensor_slices((cubos_final, z_vals_final, psd_max_final, psd_min_final, mean_psd_final))
dataset = dataset.shuffle(buffer_size = len(cubos_final)).batch(batch_size1, drop_remainder = True)


#Cargamos el Discriminador y Generador
generator = Generator_film_linear_swish(filter1 = 256, filter2 = 128, filter3 = 64)
discriminator = Discriminator_projection_swish(filter1 = 32, filter2 = 64, filter3 = 128, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size1, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder, lambda_psd_schedule = None,
                lambda_term = 20,  image_size = image_size_64, use_psd_loss = True, grad_ratio = 0.5)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
