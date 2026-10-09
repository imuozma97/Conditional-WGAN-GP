"""
Archivo para calcular el PSD de los datos originales, dependiendo de la normalización
"""

import os
import numpy as np
from preprocess_data import Dataset
from power import Power
from transforms import forward_2, backward_2
from config import batch_size1, num_cv, n_bar, image_size2



datos = Dataset(batch_size1, n_bar, buffer_size =  918)
power = Power(image_size2)

print("Cargo datos")
n_part, red = datos.load_npart("Data3D-128.hdf5")
delta = datos.delta(n_part)
print("Máx: ", np.max(delta), "Mín: ", np.min(delta))


k_values = power.compute_psd(np.squeeze(delta[1]))[1]

psd_delta = power.compute_all_psd(delta)           
psd_delta_agrupado = datos.reordenacion(num_cv, psd_delta)
                
all_mean = power.compute_all_mean(psd_delta_agrupado, num_cv)
psd_mean = np.tile(all_mean[0], (27, 1))
psd_max = np.tile(all_mean[1], (27, 1))
psd_min = np.tile(all_mean[2], (27, 1))
sigma = np.tile(all_mean[3], (27, 1))
sigma_log = np.tile(all_mean[4], (27, 1))


np.savez("PSD_delta-128.npz", psd = psd_delta, psd_agrupado = psd_delta_agrupado, mean = psd_mean, sigma = sigma, sigma_log = sigma_log, psd_max = psd_max, psd_min = psd_min, k_values = k_values)
