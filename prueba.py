from preprocess_data import Dataset
from config import batch_size1, n_bar, num_cv
import numpy as np
from transforms import forward, backward
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

datos = Dataset(batch_size1, n_bar, buffer_size=918)

#back_im = backward(images)

file = "Camels_data/Data3D-64.hdf5"
f = h5py.File(file, 'r')
maps = f['train_maps'][:]
red = np.array(f['train_labels'])[:]

percentil = np.percentile(maps, 99.99)
print("percentil: ", percentil)


maps_ordenados = datos.reordenacion(num_cv, maps)
print("Maximo para z = 0:", np.max(maps_ordenados[891:]), "Minimo para z = 0:", np.min(maps_ordenados[891:]))
print("Maximo para z = 6:", np.max(maps_ordenados[0:27]), "Minimo para z = 6:", np.min(maps_ordenados[0:27]))

#k = 8000

#rho_transf = 2*maps_ordenados/(maps_ordenados + k) -1 #Aquí ya están agrupados por redshift
rho_transf = forward(maps_ordenados)
print("Max rho:", np.max(rho_transf), "Min rho:", np.min(rho_transf)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
rho_transf_z0 = rho_transf[891:] #Cojo solo los z = 0
print("Max rho transformada z=0:", np.max(rho_transf_z0), "Min rho transformada z=0:", np.min(rho_transf_z0)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
rho_transf_z6 = rho_transf[0:27] #Cojo solo los z = 6
print("Max rho transformada z=6:", np.max(rho_transf_z6), "Min rho transformada z=6:", np.min(rho_transf_z6)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1


def histograma(data, z):
     
        values = data.numpy().flatten() if hasattr(data, "numpy") else data.flatten()

        
        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "z =  0")
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número de vóxeles")

        plt.yscale('log')

        plt.grid(False)
        plt.legend()
        plt.ylim(1, 10**7)

        
        #plt.xlim(-20, 140)
        #plt.xlim(-1, 1)
        filename = f"histo_forw_z{z}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()


histograma(rho_transf_z0, 0)
histograma(rho_transf_z6, 6)

