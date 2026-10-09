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
n_part, red = datos.load_npart("Data3D-64.hdf5")
print(red[0:34])

file = 'Sim_hidrodinamicas/Gas_positions_0.hdf5'
f = h5py.File(file, 'r')
pos = f['positions'][:]
red = np.array(f['train_labels'])

print("Posiciones gas", pos, pos.shape)
print("Redshift", red)
"""


"""
file = 'Sim_hidrodinamicas/CV_0/snapshot_014.hdf5'
f = h5py.File(file, 'r')
#print(f.keys())
pos_g = f['PartType0/Coordinates'][:]/1e3
mass = f["PartType0/Masses"]

print("MASAS", mass, mass.shape)
print(pos_g, pos_g.shape)
"""



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


#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part)
forw = forward_2(delta+1) #Esto es lo que recibe la red

#Normalizamos el redshift
z_vals = datos.factor_escala(red)

forw_agrupados = datos.reordenacion(num_cv, forw)
desnorm_data = backward_2(forw) -1 #Esto es delta
desnorm_data_agrupados = datos.reordenacion(num_cv, desnorm_data)



k_values = datos.load_k_values()
#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("PSD_delta.npz")
psd_mean_desnorm = psd_mean_desnorm[0:34]
psd_sigma_desnorm = psd_sigma_desnorm[0:34]
psd_max_desnorm = psd_max_desnorm[0:34]
psd_min_desnorm = psd_min_desnorm[0:34]


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




#SACAMOS PSD DE LOS DATOS FALSOS
"""
print("Calculando PSD de los datos falsos desnormalizados...")
psd_fake_desnorm = power.compute_all_psd(desnorm_fake_agrupados)
psd_fake_desnorm_medio = power.compute_all_mean(psd_fake_desnorm, N)
psd_fake_desnorm_mean = psd_fake_desnorm_medio[0]
psd_fake_desnorm_max = psd_fake_desnorm_medio[1]
psd_fake_desnorm_min = psd_fake_desnorm_medio[2]
psd_fake_desnorm_sigma = psd_fake_desnorm_medio[3]   

"""

#AHORA COMPARAMOS LOS PSD DE LOS DATOS REALES Y FALSOS, TANTO NORMALIZADOS COMO DESNORMALIZADOS
#print("Comparando PSD de los datos reales y falsos normalizados...")
#power.compare_psd(k_values, psd_mean_norm, psd_fake_norm_mean, psd_max_norm, psd_min_norm, psd_fake_norm_max, psd_fake_norm_min, red, generated_images_folder, f"compare_psd_norm_{epoch}", "norm")

#print("Comparando PSD de los datos reales y falsos desnormalizados...")
#power.compare_psd(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_{epoch}", "desnorm")
#power.compare_psd_residuos(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_max_desnorm, psd_min_desnorm, psd_fake_desnorm_max, psd_fake_desnorm_min, red, generated_images_folder, f"compare_psd_residuos_{epoch}", "desnorm")

#power.compare_psd_percentil(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_percentil90_{epoch}", "desnorm", N)
#power.compare_psd_percentil_residuos(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"psd_paper_{epoch}", "desnorm", N)
 
#power.compare_psd_individual(k_values, psd_mean_desnorm, psd_fake_desnorm_mean, psd_fake_desnorm, psd_max_desnorm, psd_min_desnorm, red, generated_images_folder, f"compare_psd_individual_{epoch}", "desnorm", N)


print("Sacando histogramas...")
#histogramas.all_histogramas(N, norm_fake_agrupados, forw_agrupados, "norm", epoch)
histogramas = Histogramas(generated_images_folder, red)
#print("Sacando histogramas desnormalizados...")
#histogramas.all_histogramas_medio_residuos_p90(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch)
#histogramas.all_histogramas(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch)
histogramas.all_histogramas_medio_residuos_p90(N, desnorm_fake_agrupados, desnorm_data_agrupados, "desnorm", epoch, red)



#gif(os.path.join(generated_images_folder, f"compare_psd_percentil90_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_norm_{epoch}"), f"psd_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"compare_psd_individual_{epoch}"), f"psd_gif_{epoch}.gif")

#gif(os.path.join(generated_images_folder, f"histogramas_desnormalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")
#gif(os.path.join(generated_images_folder, f"histogramas_normalizados_{epoch}"), f"histogramas_gif_{epoch}.gif")

#imagenes.save_generated_vtk(desnorm_fake, z_vals, output_folder=os.path.join(trained_models_folder, f"vtk_epoch_{epoch}"), log_scale=True)

