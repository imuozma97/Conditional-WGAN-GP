"""
Archivo para calcular el PSD de los datos originales
En caso de ser para 128, habría que cambiar n_bar, image_size y el nombre
"""

import os
import numpy as np
from preprocess_data import Dataset
from power import Power
from transforms import forward, backward
from config import batch_size, num_cv, n_bar_64, , n_bar_128, image_size_64, image_size_128

SIZE = 64 #Cambio para 128
n_bar = n_bar_64 if SIZE == 64 else n_bar_128
image_size = image_size_64 if SIZE == 64 else image_size_128

datos = Dataset(batch_size, buffer_size = 918)
power = Power(image_size_64, physical_units = True)          # normalización de entrenamiento, 31 bins

n_part, red = datos.load_npart(f"Data3D-{SIZE}.hdf5")
delta = datos.delta(n_part, n_bar_64).astype(np.float32)

psd = np.concatenate([power.compute_all_psd(delta[i:i+34]).numpy() for i in range(0, len(delta), 34)])
psd_agrupado = datos.reordenacion(num_cv, psd)
mean, pmax, pmin, sigma, sigma_log = [np.asarray(a) for a in power.compute_all_mean(psd_agrupado, num_cv)]
tile = lambda a: np.tile(a, (num_cv, 1))

np.savez(f"psd-data/PSD_delta_{SIZE}.npz", psd = psd, psd_agrupado = psd_agrupado,
         mean = tile(mean), sigma = tile(sigma), sigma_log = tile(sigma_log),
         psd_max = tile(pmax), psd_min = tile(pmin),
         k_values = power.k_centers, k_eff = power.k_eff, n_modes = power.n_modes)
         
print("Se ha guardado")

