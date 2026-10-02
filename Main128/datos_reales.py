import tensorflow as tf
import os
import numpy as np

from preprocess_data import Dataset
from generate import Fake_images
from config import batch_size1, ncritic3, n_bar2, image_size2, num_cv, num_classes
from architectures.generators128 import Generator_film_linear_swish
from architectures.discriminators128 import Discriminator_projection_swish
from training128 import Training128
from transforms import forward_2
from psd_utils import lambda_psd_schedule3
import glob


trained_models_folder = "Training128/"
generated_images_folder = "Training128/"


#Cargamos las clases necesarias
datos= Dataset(batch_size1, n_bar2, buffer_size = 918)
imagenes = Fake_images(N = num_cv, image_size = image_size2, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

#Cargamos los datos: número de partículas y redshift
print("Cargamos los datos")
n_part, red = datos.load_npart("Data3D-128.hdf5")
print("red shape", red.shape)
print("Saco delta")
delta = datos.delta(n_part)
print("redshifts")
z_vals = datos.factor_escala(red)
print("z vals shape", z_vals.shape)

print("Saco forward por partes y concateno")



for i in range(num_cv):
    forw = forward_2(delta[num_classes*i : num_classes + num_classes*i]+1)
    reds = z_vals[num_classes*i : num_classes + num_classes*i]
    imagenes.save_data(f"datos_reales_2/datos_reales_{i}.npz", forw, reds)



files = sorted(glob.glob(os.path.join(trained_models_folder, f"datos_reales_2/*.npz")))

data_list = []
labels_list = []

for file in files:
    with np.load(file) as f:
        data_list.append(f["data"])
        labels_list.append(f["labels"])

forw_final = np.concatenate(data_list, axis=0)
z_vals_final = np.concatenate(labels_list, axis=0)

imagenes.save_data("datos_reales_final_2.npz", forw_final, z_vals_final)


