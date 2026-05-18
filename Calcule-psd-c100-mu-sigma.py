"""
Archivo para calcular el PSD de los datos originales, dependiendo de la normalización
"""


import os
import numpy as np
from preprocess_data import Dataset
from power import Power
from transforms import forward, backward
from config import batch_size1, num_cv, n_bar




datos = Dataset(batch_size1)
power = Power()
#CASO: normalización con mu y sigma, pero ahora c=100, y no hay replace N voxels

images, red = datos.data0('Camels_data/Data3D-64.hdf5')
delta = datos.delta(images)
forw = forward(delta)
mu, sigma = datos.compute_mu_sigma(forw, red) #Se lo doy por evoluciones porque compute ya lo reagrupa dentro, y salen por evoluciones, como los datos
norm_data = datos.normalizar_mu_sigma(forw, mu, sigma)

k_values = power.compute_psd(norm_data[0])[1]

#Sacamos todos los psd de los datos normalizados
psd_norm = power.compute_all_psd(norm_data)           
psd_norm_agrupado = datos.reordenacion(psd_norm, red)[0]
                 
#Calculamos las medias de psd para cada redshift con los datos agrupados
all_mean_norm = power.compute_all_mean(psd_norm_agrupado, num_cv)
psd_mean_norm = np.tile(all_mean_norm[0], (27, 1))
psd_max_norm = np.tile(all_mean_norm[1], (27, 1))
psd_min_norm = np.tile(all_mean_norm[2], (27, 1))
sigma_norm = np.tile(all_mean_norm[3], (27, 1))
sigma_log_norm = np.tile(all_mean_norm[4], (27, 1))
#k_values = np.tile(k_values, (27, 1))


#np.savez("PSD_norm_mu_sigma_c100", psd = psd_norm, psd_agrupado = psd_norm_agrupado, mean = psd_mean_norm , sigma = sigma_norm,  sigma_log = sigma_log_norm, psd_max = psd_max_norm , psd_min = psd_min_norm , k_values = k_values)



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


np.savez("PSD_delta_c100", psd = psd_delta, psd_agrupado = psd_delta_agrupado, mean = psd_mean, sigma = sigma, sigma_log = sigma_log, psd_max = psd_max, psd_min = psd_min, k_values = k_values)

'''
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
'''