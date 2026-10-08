import numpy as np
from preprocess_data import Dataset
from power_dc import Power
from config import batch_size1, num_cv, n_bar_64, image_size_64

datos = Dataset(batch_size1, buffer_size = 918)
power = Power(image_size_64)

n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64).astype(np.float32)

psd = np.concatenate([power.compute_all_psd(delta[i:i+34]).numpy() for i in range(0, len(delta), 34)])
psd_agrupado = datos.reordenacion(num_cv, psd)
mean, pmax, pmin, sigma, sigma_log = [np.asarray(a) for a in power.compute_all_mean(psd_agrupado, num_cv)]
tile = lambda a: np.tile(a, (num_cv, 1))

np.savez("psd-data/PSD_delta_dc.npz", psd = psd, psd_agrupado = psd_agrupado,
         mean = tile(mean), sigma = tile(sigma), sigma_log = tile(sigma_log),
         psd_max = tile(pmax), psd_min = tile(pmin),
         k_values = power.bin_centers.numpy())
print("Guardado:", psd.shape)