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
import matplotlib.pyplot as plt

trained_models_folder = "Training3D/6-models"
generated_images_folder = "Training3D/6-images"
epoch = "00608"
N=100

datos= Dataset(batch_size1, n_bar, buffer_size = 918)
power = Power(image_size)


#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part)
forw = forward_2(delta+1) #Esto es lo que recibe la red

#Normalizamos el redshift
z_vals = datos.factor_escala(red)

forw_agrupados = datos.reordenacion(num_cv, forw)
desnorm_data = backward_2(forw) -1 #Esto es delta
desnorm_data_agrupados = datos.reordenacion(num_cv, desnorm_data)



#GENERACIÓN DE IMÁGENES FALSAS PARA LOS MEJORES PERCENTS

imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
print("Generando imágenes falsas...")
#gen_images = imagenes.generate_images(z_vals, f"best_psd_generator/epoch_{epoch}")
#imagenes.save_data(f"datos_gen_{epoch}.npz", gen_images[0], gen_images[1])


#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion(N, norm_fake)

#Desnormalizamos los datos generados

desnorm_fake = backward_2(norm_fake) -1
desnorm_fake_agrupados  = datos.reordenacion(N, desnorm_fake)

"""
slices = [0,  31, 63,]
mean_cube_fake = np.mean(desnorm_fake_agrupados[891:], axis=0)

for s in slices:
    plt.figure(figsize=(6,6))
    plt.imshow(mean_cube_fake[:, :, s], origin='lower')
    plt.title(f"Fake - slice {s}")
    plt.colorbar()
    plt.savefig(f"mean_cube_fake_slice{s}.png", dpi=300, bbox_inches="tight")
    plt.show()



mean_cube_real = np.mean(desnorm_data_agrupados[891:], axis=0)
for s in slices:
    plt.figure(figsize=(6,6))
    plt.imshow(mean_cube_real[:, :, s], origin='lower')
    plt.title(f"Real - slice {s}")
    plt.colorbar()
    plt.savefig(f"mean_cube_real_slice{s}.png", dpi=300, bbox_inches="tight")
    plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt

def distances_to_faces(cubes, percentile=99.9):
    """
    Calcula la distancia a la cara más cercana para los voxels
    por encima de un percentil dado.

    Parameters
    ----------
    cubes : ndarray (N, L, L, L)
    percentile : float

    Returns
    -------
    distances : ndarray
    """

    distances = []

    L = cubes.shape[1]

    for cube in cubes:

        # Umbral del halo
        thr = np.percentile(cube, percentile)

        # Coordenadas de los voxels más densos
        coords = np.argwhere(cube >= thr)

        for x, y, z,_ in coords:

            d = min(
                x,
                L-1-x,
                y,
                L-1-y,
                z,
                L-1-z
            )

            distances.append(d)

    return np.array(distances)



dist_real = distances_to_faces(desnorm_data_agrupados[891:], percentile=99.9)
dist_fake = distances_to_faces(desnorm_fake_agrupados[891:], percentile=99.9)


plt.figure(figsize=(7,5))

bins = np.arange(0, 33) - 0.5

plt.hist(dist_real,
         bins=bins,
         density=True,
         alpha=0.6,
         label="Real")

plt.hist(dist_fake,
         bins=bins,
         density=True,
         alpha=0.6,
         label="Fake")

plt.xlabel("Distance to nearest cube face (voxels)")
plt.ylabel("Probability")
plt.legend()
plt.savefig(f"distances_histogram.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()



for dmax in [1,2,3,4,5]:

    p_real = np.mean(dist_real <= dmax)
    p_fake = np.mean(dist_fake <= dmax)

    print(f"d <= {dmax}")
    print(f"   Real : {100*p_real:.2f}%")
    print(f"   Fake : {100*p_fake:.2f}%")