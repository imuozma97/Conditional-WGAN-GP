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
from histo import Histogramas
from cubos import cubo_part

trained_models_folder = "Results3D/1-models"
generated_images_folder = "Results3D/1-images"
epoch = "01802"

datos= Dataset(batch_size1)
power = Power()


#DATOS REALES
norm_data, z_vals, max_desnorm, min_desnorm = datos.load_data("norm")
norm_data_agrupados, _ = datos.reordenacion(norm_data, z_vals)
desnorm_data_agrupados = datos.desnormalizar_datos(norm_data_agrupados, max_desnorm, min_desnorm)
desnorm_data_agrupados = backward(desnorm_data_agrupados)

#PSD DATOS REALES NORMALIZADOS
psd_max_norm, psd_min_norm, psd_mean_norm, psd_sigma_norm = datos.load_psd("norm")
psd_mean_norm = psd_mean_norm[0:34]
psd_sigma_norm = psd_sigma_norm[0:34]
psd_max_norm = psd_max_norm[0:34]
psd_min_norm = psd_min_norm[0:34]
k_values = datos.load_k_values()

#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm = datos.load_psd("desnorm")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]



#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS

imagenes = Fake_images(N = 27, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1]) 

#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados,_ = datos.reordenacion(norm_fake, labels_fake)

#Desnormalizamos los datos generados
desnorm_fake = datos.desnormalizar_datos(norm_fake, max_desnorm, min_desnorm)
desnorm_fake = backward(desnorm_fake)

desnorm_fake_agrupados,_  = datos.reordenacion(desnorm_fake, labels_fake)


#SACAMOS PSD DE LOS DATOS FALSOS
 
print("Calculando PSD de los datos falsos normalizados...")
psd_fake_norm = power.compute_all_psd(norm_fake_agrupados)
psd_fake_norm_medio = power.compute_all_mean(psd_fake_norm, N)
psd_fake_norm_mean = psd_fake_norm_medio[0]
psd_fake_norm_max = psd_fake_norm_medio[1]
psd_fake_norm_min = psd_fake_norm_medio[2]
psd_fake_norm_sigma = psd_fake_norm_medio[3]

print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_medio[3]   


#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
print("Comparando PSD de los datos reales y falsos normalizados...")
power.compare_psd(k_values, psd_mean_norm, psd_fake_norm_mean, psd_max_norm, psd_min_norm, psd_fake_norm_max, psd_fake_norm_min, z_vals, generated_images_folder, f"compare_psd_maxmin_norm_{epoch}", "norm")
print("Comparando PSD de los datos reales y falsos desnormalizados...")
power.compare_psd(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, z_vals, generated_images_folder, f"compare_psd_maxmin_desnorm_{epoch}", "desnorm")


histogramas = Histogramas(generated_images_folder, z_vals)
print("Sacando histogramas normalizados...")
histogramas.all_histogramas(norm_fake_agrupados, norm_data_agrupados, "norm", epoch)
print("Sacando histogramas desnormalizados...")
histogramas.all_histogramas(desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch)


print("Sacando cubos de las imágenes generadas...")

cubo_part(np.log1p(desnorm_fake), z_vals, f"cubos_{epoch}", generated_images_folder)






