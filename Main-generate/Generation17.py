"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from preprocess_data import Dataset
from power_claude import Power
from config import batch_size1, image_size_64, num_cv, n_bar_64
from histo import Histogramas
from gif import gif
from transforms import forward_2,backward_2

trained_models_folder = "Training3D/17-models_dyn"
generated_images_folder = "Training3D/17-images_dyn"
epoch = "00620"
N=100

datos= Dataset(batch_size1, buffer_size = 918)
power = Power(image_size_64, physical_units=True)

indices = [4, 12, 25]
mask = np.ones(34, dtype=bool)
mask[indices] = False


#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part,n_bar_64)
forw = forward_2(delta+1) #Esto es lo que recibe la red

#Primero los datos del training
forw = forw.numpy()
cubos_final = forw.reshape(-1, 34, 64, 64, 64, 1)[:, mask]
cubos_final = cubos_final.reshape(-1, 64, 64, 64, 1)

desnorm_data = backward_2(cubos_final) -1 #Esto es delta
print("desnorm shape", desnorm_data.shape)
desnorm_data_agrupados = datos.reordenacion_nueva(num_cv, 31, desnorm_data)



#Ahora los cubos de validación
indices_todos = []

for rep in range(27):
    base = rep * 34
    indices_todos.extend([base + 4, base + 12, base + 25])

forw_val = forw[indices_todos]
desnorm_data_val = backward_2(forw_val) -1 #Esto es delta
print("shape:", desnorm_data_val.shape)
desnorm_data_agrupados_val = datos.reordenacion_nueva(num_cv, 3, desnorm_data_val)

#Normalizamos el redshift
z_vals = datos.factor_escala(red)
z_validation = z_vals[indices]

z_vals_reshape = z_vals.reshape(-1, 34)
z_training = z_vals_reshape[:, mask].reshape(-1, 1)


k_values = power.k_centers


#GENERACIÓN DE LOS DATOS DEL CONJUNTO DE VALIDACIÓN, A VER SI INTERPOLA BIEN

imagenes = Fake_images(N = N, image_size = image_size_64, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
#gen_images = imagenes.generate_images_nuevo(z_validation, f"best_psd_generator/epoch_{epoch}", 3)
#imagenes.save_data(f"datos_validation_{epoch}.npz", gen_images[0], gen_images[1])


print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_validation_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion_nueva(N, 3, norm_fake)

#Desnormalizamos los datos generados

desnorm_fake = backward_2(norm_fake) -1
desnorm_fake_agrupados  = datos.reordenacion_nueva(N, 3, desnorm_fake)


print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean_nuevo(psd_fake_desnorm, N, 3)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]


psd_real_val = power.compute_all_psd(desnorm_data_agrupados_val).numpy()
psd_mean_val = power.compute_all_mean(psd_real_val, num_cv)[0]

print("Comparando PSD de los datos reales y falsos desnormalizados...")
power.compare_psd_claude(k_values, psd_mean_val, psd_fake_desnorm, psd_real_val, red[indices], generated_images_folder, f"compare_psd_claude_{epoch}", N)
power.error_media(k_values, psd_mean_val, psd_fake_desnorm, psd_real_val, N)
power.error_dispersion(k_values, psd_fake_desnorm, psd_real_val, psd_mean_val, N)

print("Sacando histogramas ...")
histogramas = Histogramas(generated_images_folder, red[indices])
histogramas.all_histogramas_curva(N, desnorm_fake_agrupados, desnorm_data_agrupados_val, "desnorm", epoch, red[indices], f"new_compare_histo_claude_{epoch}", n_keep=None)




