"""
Archivo para quedarme con las mejores muestras, que se encuentran dentro de la banda real.
"""


import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from transforms import backward
from preprocess_data import Dataset
from power import Power
from config import batch_size1, num_classes, image_size, latent_dim, num_cv, mass, boxsize, n_bar
from histo import Histogramas
from cubos import cubo_part
from gif import gif
from psd_utils import psd_out_of_band_fraction

trained_models_folder = "Results3D/6-models"
generated_images_folder = "Results3D/6-images"
epoch = "02810"
N=100

datos= Dataset(batch_size1)
power = Power()


#DATOS REALES
#mu y sigma están ordenados por evoluciones
#_, red = datos.data0('Camels_data/Data3D-64.hdf5') y uso red en el compare_psd
norm_data, z_vals, mu, sigma = datos.load_data("redshift_norm")
desnorm_data = datos.desnormalizar_mu_sigma(norm_data, mu, sigma)

norm_data_agrupados, mu_agrupado, sigma_agrupado = datos.reordenacion(num_cv, norm_data, mu, sigma)
desnorm_data_agrupados = datos.desnormalizar_mu_sigma(norm_data_agrupados, mu_agrupado, sigma_agrupado)
desnorm_data_agrupados = backward(desnorm_data_agrupados)


#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd_real = datos.load_psd("PSD_delta_c100.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]



#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS

imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
#gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
#imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1])


#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion(N, norm_fake)
print("norm_fake_agrupados shape", norm_fake_agrupados.shape)

#Desnormalizamos los datos generados

mu_N = np.tile(mu[0:34], N)
sigma_N = np.tile(sigma[0:34], N)
desnorm_fake = datos.desnormalizar_mu_sigma(norm_fake, mu_N, sigma_N)
desnorm_fake = backward(desnorm_fake)
desnorm_fake_agrupados  = datos.reordenacion(N, desnorm_fake)


#
#SACAMOS PSD DE LOS DATOS FALSOS

print("Calculando PSD de los datos falsos desnormalizados agrupados...")
psd_fake_desnorm_agrupado = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio_agrupado = power.compute_all_mean(psd_fake_desnorm_agrupado, N)
psd_fake_desnorm_mean_agrupado = psd_fake_desnorm_medio_agrupado[0]
psd_fake_desnorm_max_agrupado = psd_fake_desnorm_medio_agrupado[1]
psd_fake_desnorm_min_agrupado = psd_fake_desnorm_medio_agrupado[2]
psd_fake_desnorm_sigma_agrupado = psd_fake_desnorm_medio_agrupado[3]   

print("Calculando PSD de los datos falsos desnormalizados por evoluciones...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake)
psd_fake_desnorm_medio = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_medio[3] 


percent = psd_out_of_band_fraction(psd_fake_desnorm[0:34], psd_min_desnorm, psd_max_desnorm)
print("Percent",percent)



