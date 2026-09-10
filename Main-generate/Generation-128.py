import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from preprocess_data import Dataset
from power import Power
from config import batch_size1, image_size2, num_cv, n_bar2, num_classes
from histo import Histogramas
from gif import gif
from transforms import forward_128,backward_128
import glob

trained_models_folder = "Training128/1-models"
generated_images_folder = "Training128/1-images"
epoch = "01030"
N=27

datos= Dataset(batch_size1, n_bar2, buffer_size = 918)
power = Power(image_size2)


#DATOS REALES
print("Cargo datos reales")
n_part, red = datos.load_npart("Data3D-128.hdf5")
delta = datos.delta(n_part)
print("delta", np.max(delta))


z_vals = datos.factor_escala(red)

data_real_agrupados = datos.reordenacion(num_cv, delta)

k_values = datos.load_k_values(image_size2)
#PSD DATOS REALES DESNORMALIZADOS 
print("PSD datos reales")
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("PSD_delta-128.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]


#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS

imagenes = Fake_images(N = N, image_size = image_size2, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando y guardando imágenes falsas tal cual ... falta hacer backward")
#gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
#imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1])

print("Cargando datos generados tal cual...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))

print("max de datos_gen", np.max(norm_fake))

"""
#Desnormalizamos los datos generados
print("Hago backward por partes")

carpeta = f"datos_gen_{epoch}"
path = os.path.join(trained_models_folder, carpeta)
if not os.path.exists(path):
    os.makedirs(path)

for i in range(N):
    desnorm_fake = backward_128(norm_fake[num_classes*i : num_classes + num_classes*i]) -1
    labels = labels_fake[num_classes*i : num_classes + num_classes*i]
    imagenes.save_data(f"datos_gen_{epoch}/datos_gen_{i}.npz", desnorm_fake, labels)
    print("labels", labels)



files = sorted(glob.glob(os.path.join(trained_models_folder, f"datos_gen_{epoch}/*.npz")))

data_list = []
labels_list = []

for file in files:
    with np.load(file) as f:
        data_list.append(f["data"])
        labels_list.append(f["labels"])

data = np.concatenate(data_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

np.savez_compressed(os.path.join(trained_models_folder, f"datos_final_{epoch}.npz"),
    data=data,
    labels=labels
)

print("data:", data.shape)
print("labels:", labels.shape)

"""
#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
desnorm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_final_{epoch}.npz"))
desnorm_fake_agrupados = datos.reordenacion(N, desnorm_fake)
print("max fake", np.max(desnorm_fake_agrupados))

"""

#SACAMOS PSD DE LOS DATOS FALSOS

print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_medio[3]   



#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
#print("Comparando PSD de los datos reales y falsos normalizados...")
#power.compare_psd(k_values, psd_mean_norm, psd_fake_norm_mean, psd_max_norm, psd_min_norm, psd_fake_norm_max, psd_fake_norm_min, red, generated_images_folder, f"compare_psd_norm_{epoch}", "norm")

print("Comparando PSD de los datos reales y falsos desnormalizados...")
#power.compare_psd(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_{epoch}", "desnorm")
#power.compare_psd_residuos(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_residuos_{epoch}", "desnorm")

#power.compare_psd_percentil(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_percentil90_{epoch}", "desnorm", N)
power.compare_psd_percentil_residuos(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_percentil_residuos_{epoch}", "desnorm", N)
 
power.compare_psd_individual(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_individual_{epoch}", "desnorm", N)


print("Sacando histogramas normalizados...")
#histogramas.all_histogramas(N, norm_fake_agrupados, forw_agrupados, "norm", epoch)
histogramas = Histogramas(generated_images_folder, red)
#print("Sacando histogramas desnormalizados...")
#histogramas.all_histogramas_medio_residuos_p90(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch)
histogramas.all_histogramas(N, desnorm_fake_agrupados, data_real_agrupados, "desnorm", epoch)
histogramas.all_histogramas_medio_residuos_p90(N, desnorm_fake_agrupados, data_real_agrupados, "desnorm", epoch, red)



#gif(os.path.join(generated_images_folder, f"compare_psd_percentil90_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_norm_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_individual_{epoch}"), f"psd_gif_{epoch}.gif")

#gif(os.path.join(generated_images_folder, f"histogramas_desnormalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"histogramas_normalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")

#imagenes.save_generated_vtk(desnorm_fake, z_vals, output_folder=os.path.join(trained_models_folder, f"vtk_epoch_{epoch}"), log_scale=True)
"""