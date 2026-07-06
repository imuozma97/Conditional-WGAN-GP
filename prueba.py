import os

from config import num_cv, num_classes, n_bar
import numpy as np
from transforms import forward_1, forward_2
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt


def histograma(data, name, z):
     
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
        filename = f"{name}_z{z}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()


def reordenacion(num_classes, muestras, *arrays):

        """
        En el caso de querer reordenar las muestras reales, muestras = num_cv, y de las flasas será N
        """
        reordered = [[] for _ in arrays]

        for j in range(num_classes):
            for i in range(muestras):

                idx = j + num_classes * i

                for k, arr in enumerate(arrays):
                    reordered[k].append(arr[idx])

        reordered = [np.array(r) for r in reordered]

        if len(reordered) == 1:
            return reordered[0]

        return tuple(reordered)

def calcule_delta(images, n_bar):
        print("Calculando delta...")
        delta = (images - n_bar)/n_bar
        return delta
#back_im = backward(images)

file = "Camels_data/Data3D-64.hdf5"
f = h5py.File(file, 'r')
maps = f['train_maps'][:]
red = np.array(f['train_labels'])[:]

#percentil = np.percentile(maps, 99.99)
#print("percentil: ", percentil)


maps_ordenados = reordenacion(num_classes, num_cv, maps)
maps_z0 = maps_ordenados[891:]
maps_z6 = maps_ordenados[0:27]
print("Maximo para z = 0:", np.max(maps_ordenados[891:]), "Minimo para z = 0:", np.min(maps_ordenados[891:]))
print("Maximo para z = 6:", np.max(maps_ordenados[0:27]), "Minimo para z = 6:", np.min(maps_ordenados[0:27]))

#print("Percentil de num particulas 99.999: ", np.percentile(maps_ordenados, 99.9))
#print("Percentil de num particulas 99.95: ", np.percentile(maps_ordenados, 99.95))
#print("Percentil de num particulas 99.99: ", np.percentile(maps_ordenados, 99.99))


delta = calcule_delta(maps_ordenados, n_bar)
delta_z0 = delta[891:]
delta_z6 = delta[0:27]
print("Delta Maximo para z = 0:", np.max(delta_z0), "Delta Minimo para z = 0:", np.min(delta_z0))
print("Delta Maximo para z = 6:", np.max(delta_z6), "Delta Minimo para z = 6:", np.min(delta_z6))



#histograma(maps_z0, "n_part", 0)
#histograma(maps_z6, "n_part", 6)

#histograma(delta_z0, "delta", 0)
#histograma(delta_z6, "delta", 6)


values = delta_z0.numpy().flatten() if hasattr(delta_z0, "numpy") else delta_z0.flatten()
        
plt.figure(figsize=(6,4))
plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "z0")
plt.xlabel("delta")
plt.ylabel("Número de vóxeles")

plt.yscale('log')

plt.grid(False)
plt.legend()
plt.ylim(0.1, 10**7)

plt.xlim(-1000, 9000)
filename = "histo_prueba.png"

plt.savefig(filename, dpi=150, bbox_inches='tight')
#plt.show()
plt.close()


#k = 8000
"""
#rho_transf = 2*maps_ordenados/(maps_ordenados + k) -1 #Aquí ya están agrupados por redshift
forw = forward_1(maps_ordenados)
print("Max rho:", np.max(forw), "Min rho:", np.min(forw)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
forw_z0 = forw[891:] #Cojo solo los z = 0
print("Max rho transformada z=0:", np.max(forw_z0), "Min rho transformada z=0:", np.min(forw_z0)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
forw_z6 = forw[0:27] #Cojo solo los z = 6
print("Max rho transformada z=6:", np.max(forw_z6), "Min rho transformada z=6:", np.min(forw_z6)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1


print("calculando forward delta")
forw_delta = forward_2(delta+1)        
print("Max delta:", np.max(forw_delta), "Min delta:", np.min(forw_delta)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
forw_delta_z0 = forw_delta[891:] #Cojo solo los z = 0
print("Max delta transformada z=0:", np.max(forw_delta_z0), "Min delta transformada z=0:", np.min(forw_delta_z0)) #Esto es para comprobar que la transformación se ha hecho bien, el rango es de -1 a 1
forw_delta_z6 = forw_delta[0:27] #Cojo solo los z = 6
print("Max delta transformada z=6:", np.max(forw_delta_z6), "Min delta transformada z=6:", np.min(forw_delta_z6))


histograma(forw_z0, "forw", 0)
histograma(forw_z6, "forw", 6)

histograma(forw_delta_z0, "forw_delta", 0)
histograma(forw_delta_z6, "forw_delta", 6)

"""