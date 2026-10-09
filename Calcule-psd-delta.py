"""
Archivo para calcular el PSD de los datos originales, dependiendo de la normalización
"""

import os
import numpy as np
from preprocess_data import Dataset
from power_claude import Power
from transforms import forward_2, backward_2
from config import batch_size1, num_cv, n_bar_64, image_size_64



datos = Dataset(batch_size1, buffer_size = 918)
power = Power(image_size_64)          # normalización de entrenamiento, 31 bins

n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64).astype(np.float32)

psd = np.concatenate([power.compute_all_psd(delta[i:i+34]).numpy() for i in range(0, len(delta), 34)])
psd_agrupado = datos.reordenacion(num_cv, psd)
mean, pmax, pmin, sigma, sigma_log = [np.asarray(a) for a in power.compute_all_mean(psd_agrupado, num_cv)]
tile = lambda a: np.tile(a, (num_cv, 1))

np.savez("psd-data/PSD_delta_claude.npz", psd = psd, psd_agrupado = psd_agrupado,
         mean = tile(mean), sigma = tile(sigma), sigma_log = tile(sigma_log),
         psd_max = tile(pmax), psd_min = tile(pmin),
         k_values = power.k_centers, k_eff = power.k_eff, n_modes = power.n_modes)


"""

datos = Dataset(batch_size1, buffer_size =  918)
power = Power(image_size_64)

print("Cargo datos")
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64).astype(np.float32)

print("PSD")
#Sacamos todos los psd de los datos normalizados

psd_norm = power.compute_all_psd(norm)      
psd_norm_agrupado = datos.reordenacion(num_cv, psd_norm)
#Calculamos las medias de psd para cada redshift con los datos agrupados
all_mean_norm = power.compute_all_mean(psd_norm_agrupado, num_cv)
psd_mean_norm = np.tile(all_mean_norm[0], (27, 1))
psd_max_norm = np.tile(all_mean_norm[1], (27, 1))
psd_min_norm = np.tile(all_mean_norm[2], (27, 1))


np.savez("PSD_forw_norm.npz", psd = psd_norm, psd_agrupado = psd_norm_agrupado, mean = psd_mean_norm , sigma = sigma_norm,  sigma_log = sigma_log_norm, psd_max = psd_max_norm , psd_min = psd_min_norm , k_values = k_values)



#CASO DESNORMALIZADO

#Habría que deshacerlo de la misma forma. Si hago psd de delta puede cambiar un poco de si hago los mismos pasos que en el generador
forw_data = datos.desnormalizar_mu_sigma(norm_data, mu, sigma)
delta_data = backward(forw_data)

psd_delta = power.compute_all_psd(delta_data)           
psd_delta_agrupado = datos.reordenacion(psd_delta, data[1])[0]
                
all_mean = power.compute_all_mean(psd_delta_agrupado, num_cv)
psd_mean = np.tile(all_mean[0], (27, 1))
psd_max = np.tile(all_mean[1], (27, 1))
psd_min = np.tile(all_mean[2], (27, 1))
sigma = np.tile(all_mean[3], (27, 1))
sigma_log = np.tile(all_mean[4], (27, 1))


np.savez("PSD_delta_c100_128", psd = psd_delta, psd_agrupado = psd_delta_agrupado, mean = psd_mean, sigma = sigma, sigma_log = sigma_log, psd_max = psd_max, psd_min = psd_min, k_values = k_values)


part_data = datos.deshacer_delta(delta_data)
psd_part = power.compute_all_psd(part_data)           
psd_part_agrupado = datos.reordenacion(psd_part, data[1], num_cv)[0]
                
all_mean_part = power.compute_all_mean(psd_part_agrupado, num_cv)
psd_mean_part = np.tile(all_mean_part[0], (27, 1))
psd_max_part = np.tile(all_mean_part[1], (27, 1))
psd_min_part = np.tile(all_mean_part[2], (27, 1))
sigma_part = np.tile(all_mean_part[3], (27, 1))
sigma_log_part = np.tile(all_mean_part[4], (27, 1))

np.savez("PSD_part_c100", psd = psd_part, psd_agrupado = psd_part_agrupado, mean = psd_mean_part, sigma = sigma_part, sigma_log = sigma_log_part, psd_max = psd_max_part, psd_min = psd_min_part, k_values = k_values)
"""