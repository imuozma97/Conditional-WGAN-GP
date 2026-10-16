"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from preprocess_data import Dataset
from power import Power
from config import batch_size, image_size_64, num_cv, n_bar_64
from histo import Histogramas
from gif import gif
from transforms import forward, backward
from coherencia_temp import coherence


trained_models_folder = "Results-64/baseline-models"
generated_images_folder = "Results-64/baseline-images"
epoch = "00683"
N = 100

datos= Dataset(batch_size, buffer_size = 918)
power = Power(image_size_64, physical_units = True)


#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64)
#Normalizamos el redshift
z_vals = datos.factor_escala(red)

desnorm_data_agrupados = datos.reordenacion(num_cv, delta)


k_values = power.k_centers
#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("psd-data/PSD_delta_64.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]


#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS

imagenes = Fake_images(N = N, image_size = image_size_64, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1])


#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion(N, norm_fake)

desnorm_fake = []
for i in range(len(norm_fake)):
    desnorm_fake.append(backward(norm_fake[i]) -1)

desnorm_fake = np.array(desnorm_fake)
desnorm_fake_agrupados  = datos.reordenacion(N, desnorm_fake)



print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_medio[3]   

print("Calculo psd reales para agruparlos")
psd_real_desnorm = power.compute_all_psd(desnorm_data_agrupados)


#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
print("Comparando PSD de los datos reales y falsos ...")
power.compare_psd_final(k_values, psd_mean_desnorm, psd_fake_desnorm, psd_real_desnorm, red, generated_images_folder, f"compare_psd_{epoch}", N)

power.error_media(k_values, psd_mean_desnorm, psd_fake_desnorm, psd_real_desnorm,  N)
power.error_dispersion(k_values, psd_fake_desnorm, psd_real_desnorm, psd_mean_desnorm, N)


print("Sacando histogramas ...")
histogramas = Histogramas(generated_images_folder, red)
histogramas.all_histogramas_curva(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch, red, f"compare_histo_{epoch}")


#print("Comparamos coherencia temporal")
coherence(desnorm_data, desnorm_fake, image_size = image_size_64, N = N)


imagenes.save_generated_vtk(desnorm_fake, z_vals, output_folder=os.path.join(trained_models_folder, f"vtk_epoch_{epoch}"), log_scale=True)


