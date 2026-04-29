"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from transforms import backward
from preprocess_data import Dataset
from power import Power
from config import batch_size1, num_classes, image_size, latent_dim, num_cv, mass, boxsize, n_bar, N

trained_models_folder = "Results3D/2-models"
generated_images_folder = "Results3D/2-images"

epoch = "00900"


#DATOS REALES

datos= Dataset(batch_size1)
norm_data, z_vals, max_desnorm, min_desnorm = datos.load_data("norm")
norm_data_agrupados = datos.reordenacion(norm_data, z_vals)

#PSD DATOS REALES NORMALIZADOS
load_psd_norm = datos.load_psd("psd-data/PSD_norm.npz")
psd_mean_norm = load_psd_norm["mean"][0:34]
psd_sigma_norm = load_psd_norm["sigma_log"][0:34]
psd_max_norm = load_psd_norm["psd_max"][0:34]
psd_min_norm = load_psd_norm["psd_min"][0:34]
k_values = load_psd_norm["k_values"]

#PSD DATOS REALES DESNORMALIZADOS 
load_psd_desnorm = datos.load_psd("psd-data/PSD_desnorm.npz")
psd_mean_desnorm = load_psd_desnorm["mean"][0:34]
psd_sigma_desnorm = load_psd_desnorm["sigma_log"][0:34]
psd_max_desnorm = load_psd_desnorm["psd_max"][0:34]
psd_min_desnorm = load_psd_desnorm["psd_min"][0:34]





#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS
imagenes = Fake_images(N = 27, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

gen_images = imagenes.generate_images(z_vals, f"best_percent_generator/epoch_{epoch}")
imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1]) 

#Cargamos los datos generados para calcular espectros
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion(norm_fake, labels_fake)

#Desnormalizamos los datos generados
desnorm_fake_agrupados = datos.desnormalizar(norm_fake_agrupados, max_desnorm, min_desnorm)
desnorm_fake_agrupados = backward(desnorm_fake_agrupados)


#SACAMOS PSD DE LOS DATOS FALSOS
power = Power()
psd_fake_norm = power.compute_all_psd(norm_fake_agrupados)
psd_fake_norm_mean = power.compute_all_mean(psd_fake_norm, N)
psd_fake_norm_mean = psd_fake_norm_mean[0]
psd_fake_norm_max = psd_fake_norm_mean[1]
psd_fake_norm_min = psd_fake_norm_mean[2]
psd_fake_norm_sigma = psd_fake_norm_mean[3]

psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_mean = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_mean[0]
psd_fake_desnorm_max = psd_fake_desnorm_mean[1]
psd_fake_desnorm_min = psd_fake_desnorm_mean[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_mean[3]   


#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
power.compare_psd(k_values, psd_mean_norm, psd_fake_norm_mean, psd_max_norm, psd_min_norm, psd_fake_norm_max, psd_fake_norm_min, "compare_psd_maxmin_norm", "norm")
power.compare_psd(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, "compare_psd_maxmin_desnorm", "desnorm")















#gen_images_psd = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch_psd}")
#imagenes.save_data(f"datos_gen_{epoch_psd}.npz", gen_images_psd[0], gen_images_psd[1]) 
