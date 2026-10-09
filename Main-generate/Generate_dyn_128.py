import tensorflow as tf
import os
import numpy as np
import glob

from generate import Fake_images
from preprocess_data import Dataset
from power_claude import Power
from config import batch_size1, image_size_128, num_cv, n_bar_128, num_classes
from histo import Histogramas
from transforms import backward_128         # LA MISMA inversa que en training128_dyn.py (línea 86)

trained_models_folder = "Training128/dyn-models"
generated_images_folder = "Training128/dyn-images"
epoch = "00663"
N = 27
TROZO = 34

datos = Dataset(batch_size1, buffer_size = 918)
power = Power(image_size_128)


def psd_por_trozos(x):
    """P(k) por trozos: no convierte todos los cubos en un único tensor."""
    return np.concatenate([power.compute_all_psd(x[i:i + TROZO]).numpy() for i in range(0, len(x), TROZO)])


#DATOS REALES
print("Cargo datos reales")
n_part, red = datos.load_npart("Data3D-128.hdf5")
z_vals = datos.factor_escala(red)
delta = datos.delta(n_part, n_bar_128)
del n_part
data_real_agrupados = datos.reordenacion(num_cv, delta)
del delta

k_values = power.k_centers

#PSD DATOS REALES
print("PSD datos reales")
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("psd-data/PSD_delta_claude_128.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]

psd_real_desnorm = psd_por_trozos(data_real_agrupados)


#GENERACIÓN DE IMÁGENES FALSAS
imagenes = Fake_images(N = N, image_size = image_size_128, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder)

print("Generando y guardando imágenes falsas tal cual ... falta hacer backward")
gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1])
del gen_images

print("Cargando datos generados tal cual...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))


#Desnormalizamos los datos generados por partes (una evolución de 34 redshifts cada vez)
print("Hago backward por partes")
carpeta = f"datos_gen_{epoch}"
os.makedirs(os.path.join(trained_models_folder, carpeta), exist_ok = True)

for i in range(N):
    desnorm_trozo = backward_128(norm_fake[num_classes*i : num_classes + num_classes*i]).numpy() - 1
    labels = labels_fake[num_classes*i : num_classes + num_classes*i]
    imagenes.save_data(f"{carpeta}/datos_gen_{i:03d}.npz", desnorm_trozo, labels)
del norm_fake

files = sorted(glob.glob(os.path.join(trained_models_folder, f"{carpeta}/*.npz")))
data_list, labels_list = [], []
for file in files:
    with np.load(file) as f:
        data_list.append(f["data"])
        labels_list.append(f["labels"])
desnorm_fake = np.concatenate(data_list, axis = 0)
labels_fake = np.concatenate(labels_list, axis = 0)
del data_list, labels_list
print("data:", desnorm_fake.shape, "| labels:", labels_fake.shape)

desnorm_fake_agrupados = datos.reordenacion(N, desnorm_fake)
del desnorm_fake


print("Calculando PSD de los datos falsos...")
psd_fake_desnorm = psd_por_trozos(desnorm_fake_agrupados)


#COMPARACIÓN DE LOS PSD
print("Comparando PSD de los datos reales y falsos ...")
power.compare_psd_claude(k_values, psd_mean_desnorm, psd_fake_desnorm, psd_real_desnorm, red, generated_images_folder, f"compare_psd_claude_percentil_{epoch}", N)

#ERRORES DEL PSD
power.error_media(k_values, psd_mean_desnorm, psd_fake_desnorm, psd_real_desnorm, N)
power.error_dispersion(k_values, psd_fake_desnorm, psd_real_desnorm, psd_mean_desnorm, N)


print("Sacando histogramas ...")
histogramas = Histogramas(generated_images_folder, red)
histogramas.all_histogramas_claude(N, desnorm_fake_agrupados, data_real_agrupados, "desnorm", epoch, red, f"compare_histo_claude_{epoch}")


#Coherencia temporal: pendiente de adaptar a 128^3 (coherencia_temp.py no está en el repositorio)
#coherence(data_real, desnorm_fake, image_size = image_size_128, N = N)