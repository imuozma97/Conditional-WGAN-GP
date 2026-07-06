"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from preprocess_data import Dataset
from power import Power
from config import batch_size1, image_size, num_cv, n_bar
from histo import Histogramas
from gif import gif
from transforms import forward_2,backward_2

trained_models_folder = "Training3D/17-models"
generated_images_folder = "Training3D/17-images"
epoch = "01823"
N=100

datos= Dataset(batch_size1, n_bar, buffer_size = 918)
power = Power(image_size)

indices = [4, 12, 25]
mask = np.ones(34, dtype=bool)
mask[indices] = False


#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part)
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






k_values = datos.load_k_values()


#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("PSD_delta.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]

psd_max_val = psd_max_desnorm[indices]
psd_min_val = psd_min_desnorm[indices]
psd_mean_val = psd_mean_desnorm[indices]

psd_max_reshape = psd_max_desnorm.reshape(-1, 34, 32)
psd_max_training = psd_max_reshape[:, mask].reshape(-1, 32)

psd_min_reshape = psd_min_desnorm.reshape(-1, 34, 32)
psd_min_training = psd_min_reshape[:, mask].reshape(-1, 32)

mean_psd_reshape = psd_mean_desnorm.reshape(-1, 34, 32)
mean_psd_training = mean_psd_reshape[:, mask].reshape(-1, 32)




"""
#GENERACIÓN DE IMÁGENES FALSAS 

imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
gen_images = imagenes.generate_images_nuevo(z_training, f"best_psd_generator/epoch_{epoch}", 31)
imagenes.save_data(f"datos_training_{epoch}.npz", gen_images[0], gen_images[1])


#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_training_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion_nueva(N, 31, norm_fake)

#Desnormalizamos los datos generados

desnorm_fake = backward_2(norm_fake) -1
desnorm_fake_agrupados  = datos.reordenacion_nueva(N, 31, desnorm_fake)



print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean_nuevo(psd_fake_desnorm, N, 31)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]   



#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
#print("Comparando PSD de los datos reales y falsos normalizados...")
#power.compare_psd(k_values, psd_mean_norm, psd_fake_norm_mean, psd_max_norm, psd_min_norm, psd_fake_norm_max, psd_fake_norm_min, red, generated_images_folder, f"compare_psd_norm_{epoch}", "norm")

print("Comparando PSD de los datos reales y falsos desnormalizados...")
#power.compare_psd(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_{epoch}", "desnorm")
#power.compare_psd_residuos(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_residuos_{epoch}", "desnorm")

#power.compare_psd_percentil(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_percentil90_{epoch}", "desnorm", N)
power.compare_psd_percentil_residuos(k_values, mean_psd_training, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_training, psd_min_training, red, generated_images_folder, f"compare_psd_training_p90_{epoch}", "desnorm", N)
 
#power.compare_psd_individual(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_individual_{epoch}", "desnorm", N)



#3print("Sacando histogramas normalizados...")
#histogramas.all_histogramas(N, norm_fake_agrupados, forw_agrupados, "norm", epoch)
#print("Sacando histogramas desnormalizados...")
#histogramas.all_histogramas(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch)
print("Sacando histogramas desnormalizados...")
histogramas = Histogramas(generated_images_folder, red)
histogramas.all_histogramas_medio_p90_nuevo(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch, 31)




#gif(os.path.join(generated_images_folder, f"compare_psd_desnorm_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_norm_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_individual_{epoch}"), f"psd_gif_{epoch}.gif")

#gif(os.path.join(generated_images_folder, f"histogramas_desnormalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"histogramas_normalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")

#imagenes.save_generated_vtk(desnorm_fake, z_vals, output_folder=os.path.join(trained_models_folder, f"vtk_epoch_{epoch}"), log_scale=True)




"""
#GENERACIÓN DE LOS DATOS DEL CONJUNTO DE VALIDACIÓN, A VER SI INTERPOLA BIEN

imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
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


print("Comparando PSD de los datos reales y falsos desnormalizados...")
power.compare_psd_percentil_residuos(k_values, psd_mean_val, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_val, psd_min_val, red[indices], generated_images_folder, f"compare_psd_validation_{epoch}", "desnorm", N)
 
print("Sacando histogramas ...")
histogramas = Histogramas(generated_images_folder, red)
histogramas.all_histogramas_medio_p90_nuevo(N, desnorm_fake_agrupados, desnorm_data_agrupados_val, "desnorm", epoch, red[indices], 3)






