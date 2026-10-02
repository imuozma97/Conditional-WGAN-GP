"""
Aquí pongo la clase Dataset, que es la que contiene las funciones necesarias para el preprocesamiento de los datos antes del entrenamiento
"""

import tensorflow as tf
import numpy as np
import h5py
import os

from config import num_classes, num_cv

class Dataset(tf.keras.Model):
    def __init__(self, batch_size, buffer_size):
        super().__init__()
        self.batch_size = batch_size #Tiene que ser input porque no siempre es el mismo
        self.buffer_size = buffer_size

        
    def data0(self, file):
        """
        El archivo Data3D-64.hdf5 sería el que hay que usar
        En este caso, los datos están ordenados por evoluciones, desde z=6 a z=0
        """
        f = h5py.File(file, 'r')
        maps = f['train_maps'][:]
        red = np.array(f['train_labels'])[:]
        maps = np.expand_dims(maps, -1)
        
        return maps, red
    
    
    
    def delta(self, images, n_bar):
        
        delta = (images - n_bar)/n_bar
        #delta = np.expand_dims(delta, -1)
        return delta

    def deshacer_delta(self, delta, n_bar):
        images = delta * n_bar + n_bar
        return images


    

    def normalizar_z(self, redshifts):
        return (redshifts - np.min(redshifts))/(np.max(redshifts)- np.min(redshifts)).astype("float32")
    
    def factor_escala(self, redshifts):
        return 1/(1+redshifts).astype("float32")
    

    
    def crea_dataset(self,  *data):
    
      dataset = tf.data.Dataset.from_tensor_slices(data)
      dataset = dataset.shuffle(buffer_size = self.buffer_size).batch(self.batch_size)
  
      return dataset

    def load_npart(self, file):
        output = os.path.join("Camels_data", file)
        images, red = self.data0(output)

        return images, red

    def transform_npart(self, images, k):
        rho_transf = 2*images/(images + k) -1 #Aquí ya están agrupados por redshift
        rho_transf = np.expand_dims(rho_transf, -1)
        return rho_transf

    def inverse_transform_npart(self, images, k):
        rho_original = k * (1 + images) / (1 - images)
        rho_original = np.expand_dims(rho_original, -1)
        return rho_original


    def load_psd(self, psd_file):

        load_psd = np.load(psd_file)
        all_psd = load_psd["psd"]
        psd_mean = load_psd["mean"]
        psd_sigma = load_psd["sigma_log"]
        psd_max = load_psd["psd_max"]
        psd_min = load_psd["psd_min"]

        return psd_max, psd_min, psd_mean, psd_sigma, all_psd
    

    def load_k_values(self, image_size):
        if image_size == 64:
            load_psd = np.load("psd-data/PSD_delta.npz")
        if image_size == 128:
            load_psd = np.load("psd-data/PSD_delta_128.npz")
        k_values = load_psd["k_values"]

        return k_values


    def reordenacion(self, muestras, *arrays):

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


    def reordenacion_nueva(self, muestras, n_classes, *arrays):

        """
        En el caso de querer reordenar las muestras reales, muestras = num_cv, y de las flasas será N
        """
        reordered = [[] for _ in arrays]

        for j in range(n_classes):
            for i in range(muestras):

                idx = j + n_classes * i

                for k, arr in enumerate(arrays):
                    reordered[k].append(arr[idx])

        reordered = [np.array(r) for r in reordered]

        if len(reordered) == 1:
            return reordered[0]

        return tuple(reordered)



    def ordenar_datos_evoluciones(self, images, redshifts):
        
        """
        Para normalizar los datos, primero necesitamos que estén ordenados por evoluciones
        
        """
        
        order_images = []
        order_redshifts = []
        
        for j in range(self.num_cv):
            for i in range(self.num_classes):
                order_images.append(images[j + self.num_cv*i])
                order_redshifts.append(redshifts[j + self.num_cv*i])
        order_images = np.array(order_images)
        order_images = np.reshape(order_images, (-1, self.image_size, self.image_size, 1)) 

        order_redshifts = np.array(order_redshifts)
        order_redshifts = np.squeeze(order_redshifts)
        
        return order_images, order_redshifts

