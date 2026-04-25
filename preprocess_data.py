"""
Aquí pongo la clase Dataset, que es la que contiene las funciones necesarias para el preprocesamiento de los datos antes del entrenamiento
"""

import tensorflow as tf
import numpy as np
import h5py

from config import n_bar, buffer_size
from transforms import forward

class Dataset(tf.keras.Model):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size #Tiene que ser input porque no siempre es el mismo

        
    def data0(self, file):
        """
        El archivo Data3D-64.hdf5 sería el que hay que usar
        En este caso, los datos están ordenados por evoluciones, desde z=6 a z=0
        """
        f = h5py.File(file, 'r')
        maps = f['train_maps'][:]
        red = np.array(f['train_labels'])[:]
        
        return maps, red
    
    
    
    def delta(self, images):
        
        delta = (images - n_bar)/n_bar
        return delta
    
    
    def replace_extreme_voxels(self, data, quit=20): #Pruebo a quitar más??

        data_new = data.copy()
        flat = data_new.ravel()

        # índices de los N valores más grandes
        idx = np.argpartition(flat, -quit)[-quit:]

        # convertir a coordenadas (cubo,x,y,z)
        coords = np.array(np.unravel_index(idx, data_new.shape)).T

        for cube, x, y, z in coords:

            field = data_new[cube]

            # vecinos 3x3x3
            x0, x1 = max(x-1,0), min(x+2, field.shape[0])
            y0, y1 = max(y-1,0), min(y+2, field.shape[1])
            z0, z1 = max(z-1,0), min(z+2, field.shape[2])

            neighborhood = field[x0:x1, y0:y1, z0:z1]

            # eliminar el voxel central
            neighbors = neighborhood.flatten()
            center_index = (x-x0)*((y1-y0)*(z1-z0)) + (y-y0)*(z1-z0) + (z-z0)
            neighbors = np.delete(neighbors, center_index)

            neighbor_mean = np.mean(neighbors)

            data_new[cube, x, y, z] = neighbor_mean

        return data_new

    
    
    def normalizar_datos(self, images):

        min_val = np.min(images)
        max_val = np.max(images)
            
        normalized_data = 2 * (images - min_val) / (max_val - min_val) - 1
        normalized_data = np.expand_dims(normalized_data, -1)

        return normalized_data, max_val, min_val

    def normalizar_z(self, redshifts):
        return (redshifts - np.min(redshifts))/(np.max(redshifts)- np.min(redshifts)).astype("float32")
    

    
    def crea_dataset(self, *data):
    
      dataset = tf.data.Dataset.from_tensor_slices(data)
      dataset = dataset.shuffle(buffer_size=buffer_size).batch(self.batch_size)
  
      return dataset


    def load_data(self, data_mode):

        images, red = self.data0('../Camels_data/Data3D-64.hdf5')
        images_clean = self.replace_extreme_voxels(images, quit=20) #Quito los 20 valores extremos
        delta = self.delta(images_clean)
        forw = forward(delta)

        z_vals = self.normalizar_z(red)

        if data_mode == "norm":
            norm_data, max_desnorm, min_desnorm = self.normalizar_datos(forw)
            return norm_data, z_vals, max_desnorm, min_desnorm 

        elif data_mode == "desnorm":
            return forw, z_vals

        else:
            raise ValueError("data_mode debe ser 'norm' o 'desnorm'")



    def load_psd(self, psd_mode):

        if psd_mode == "desnorm":
            psd_file = "psd-data/PSD_desnorm.npz"
        elif psd_mode == "norm":
            psd_file = "psd-data/PSD_norm.npz"
        else:
            raise ValueError("psd_mode debe ser 'norm' o 'desnorm'")

        load_psd = np.load(psd_file)
        psd_mean = load_psd["mean"]
        psd_sigma = load_psd["sigma_log"]
        psd_max = load_psd["psd_max"]
        psd_min = load_psd["psd_min"]

        return psd_max, psd_min, psd_mean, psd_sigma
          
