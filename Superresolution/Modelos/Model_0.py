import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_XLA_ENABLE'] = '0'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from preprocess_data_sr import Dataset_sr
from config_sr import batch_size, ncritic3, boxsize
from Architectures_sr.generator_sr import Generator_sr
from Architectures_sr.discriminator_sr import Discriminator_sr
from training_dyn_sr import Training
from transforms_sr import forward

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


trained_models_folder = "Results_SR/0-models"
generated_images_folder = "Results_SR/0-images"
os.makedirs(generated_images_folder, exist_ok = True)
os.makedirs(trained_models_folder, exist_ok = True)

#Cargamos las clases necesarias
datos = Dataset_sr(batch_size, buffer_size = 200)

n_part_LR, n_part_HR, red = datos.load_pares("mallas/fiducial", "mallas/fiducial_HR", indices = range(0, 80), n_subcubos = 4, lado_LR = 64, semilla = 0)

n_bar_LR = datos.n_bar("mallas/fiducial", agrupar = 1, indice = 0)
n_bar_HR = datos.n_bar("mallas/fiducial_HR", agrupar = 1, indice = 0)

delta_LR = datos.delta(n_part_LR, n_bar_LR)
delta_HR = datos.delta(n_part_HR, n_bar_HR)
del n_part_LR, n_part_HR

forw_LR = forward(delta_LR + 1)
forw_HR = forward(delta_HR + 1)
del delta_LR, delta_HR

z_vals = datos.factor_escala(red)

dataset = datos.crea_dataset(forw_LR, forw_HR, z_vals)


#Cargamos el Discriminador y Generador
generator = Generator_sr(filter1 = 64, filter2 = 32, filter3 = 16)
discriminator = Discriminator_sr(filter1 = 16, filter2 = 32, filter3 = 64, layer = "F")


#Cargamos la red principal
cgan = Training(data_class = datos, discriminator = discriminator, generator = generator, batch_size = batch_size, ncritic = ncritic3, 
                trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder,
                lambda_term = 20, margen = 4, boxsize = boxsize, use_psd_loss = True, grad_ratio = 0.5, warmup_epochs = 0)
cgan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005, beta_1 = 0, beta_2 = 0.9),
             g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0, beta_2 = 0.9))


cgan.train(dataset, epochs = 20000)
