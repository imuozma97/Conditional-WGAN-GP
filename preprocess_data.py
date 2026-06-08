"""
Aquí pongo la clase Dataset, que es la que contiene las funciones necesarias para el preprocesamiento de los datos antes del entrenamiento
"""

import tensorflow as tf
import numpy as np
import h5py
import os

from config import num_classes, num_cv
from transforms import forward, forward_custom

class Dataset(tf.keras.Model):
    def __init__(self, batch_size, n_bar, buffer_size):
        super().__init__()
        self.batch_size = batch_size #Tiene que ser input porque no siempre es el mismo
        self.n_bar = n_bar
        self.buffer_size = buffer_size

        
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
        
        delta = (images - self.n_bar)/self.n_bar
        delta = np.expand_dims(delta, -1)
        return delta

    def deshacer_delta(self, delta):
        images = delta * self.n_bar + self.n_bar
        return images


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

    
    
    def normalizar_datos_tanh(self, images):

        min_val = np.min(images)
        max_val = np.max(images)
            
        normalized_data = 2 * (images - min_val) / (max_val - min_val) - 1
        normalized_data = np.expand_dims(normalized_data, -1)

        return normalized_data, max_val, min_val

    def normalizar_z(self, redshifts):
        return (redshifts - np.min(redshifts))/(np.max(redshifts)- np.min(redshifts)).astype("float32")
    
    def factor_escala(self, redshifts):
        return 1/(1+redshifts).astype("float32")


    def desnormalizar_datos_tanh(self, images, maximo, minimo):
        
        original_data =  ((images + 1) / 2) * (maximo - minimo) + minimo
        #original_data =  images * (maximo - minimo) + minimo
        
        return original_data
    

    
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


    def rotation(self, cube, k):
        # Rotar 90 grados en el plano de los dos primeros ejes espaciales (ejes 1 y 2)
        # k=1 significa 90°, k=2 es 180°, k=3 es 270°
        cubos_rotados = np.rot90(cube, k , axes=(1, 2))
        return cubos_rotados




    def load_data(self, file, data_mode):

        output = os.path.join("Camels_data", file)
        images, red = self.data0(output)
        delta = self.delta(images)
        forw = forward(delta)
    
        #z_vals = self.normalizar_z(red)
        z_vals = self.factor_escala(red)


        if data_mode == "global_norm_tanh":
            norm_data, max_desnorm, min_desnorm = self.normalizar_datos_tanh(forw)
            return norm_data, z_vals, max_desnorm, min_desnorm 

        if data_mode == "global_norm_tanh_new":
            forw_new = forward_custom(delta)
            forw_new = np.expand_dims(forw_new, -1)
            return forw_new, z_vals

        elif data_mode == "redshift_norm":
            mu, sigma = self.compute_mu_sigma(forw) #Se lo doy por evoluciones porque compute ya lo reagrupa dentro, y salen por evoluciones, como los datos
            norm_data = self.normalizar_mu_sigma(forw, mu, sigma)
            return norm_data, z_vals, mu, sigma

        elif data_mode == "forw": #Este para el caso linear sin hacer la "norm" de mu y sigma
            forw = np.expand_dims(forw, -1)
            return forw, z_vals


        #Trrndría que añadir otro más en caso de querer hacer ambas cosas; hacer la norm de redshift y luego tanh

        else:
            raise ValueError("Elige bien el data_mode, 'global_norm' o 'redshift_norm'")



    def load_psd(self, psd_file):

        load_psd = np.load(psd_file)
        all_psd = load_psd["psd"]
        psd_mean = load_psd["mean"]
        psd_sigma = load_psd["sigma_log"]
        psd_max = load_psd["psd_max"]
        psd_min = load_psd["psd_min"]

        return psd_max, psd_min, psd_mean, psd_sigma, all_psd
    
    def load_k_values(self):
        load_psd = np.load("psd-data/PSD_norm.npz")
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




    def compute_mu_sigma(self, snap):

        snap_ordenado = self.reordenacion(num_cv, snap)
        mu = []
        sigma = []
        for i in range(num_classes):
            mu.append(np.mean(snap_ordenado[i*num_cv:(i+1)*num_cv]))
            sigma.append(np.std(snap_ordenado[i*num_cv:(i+1)*num_cv]))

        mu = np.array(mu)
        sigma = np.array(sigma)

        mu_expanded = np.tile(mu, num_cv)
        sigma_expanded = np.tile(sigma, num_cv)

        return mu_expanded, sigma_expanded




    def normalizar_mu_sigma(self, images, mu, sigma):

        mu = tf.reshape(mu, (-1, 1, 1, 1))
        sigma = tf.reshape(sigma, (-1, 1, 1, 1))

        normalized_data = (images - mu) / sigma
        normalized_data = np.expand_dims(normalized_data, -1)

        return normalized_data

    def desnormalizar_mu_sigma(self, images, mu, sigma):

        mu = tf.reshape(mu, (-1, 1, 1, 1))
        sigma = tf.reshape(sigma, (-1, 1, 1, 1))
        images = np.squeeze(images, axis=-1)
        print("images shape", images.shape)
        print("mu shape", mu.shape)
        print("sigma shape", sigma.shape)
        original_data = images * sigma + mu
        original_data = np.expand_dims(original_data, -1)

        return original_data    



    def load_data_new(self, data_mode, salida = None):

        images, red = self.data0('Camels_data/Data3D-64.hdf5')
        #images_clean = self.replace_extreme_voxels(images, quit=20) #Quito los 20 valores extremos
        delta = self.delta(images_clean)
        forw = forward(delta)

        z_vals = self.normalizar_z(red)

        mu, sigma = self.compute_mu_sigma(forw) #Se lo doy por evoluciones porque compute ya lo reagrupa dentro, y salen por evoluciones, como los datos

        if data_mode == "norm":
            if salida == "tanh": 
                norm_data = self.normalizar_mu_sigma(forw, mu, sigma)
                min_val = np.min(norm_data)
                max_val = np.max(norm_data)
                norm_data = 2 * (norm_data - min_val) / (max_val - min_val) - 1

                return norm_data, z_vals, mu, sigma, min_val, max_val


            if salida == "sigmoid":
                norm_data = self.normalizar_mu_sigma(forw, mu, sigma)
                min_val = np.min(norm_data)
                max_val = np.max(norm_data)
                norm_data = (norm_data - min_val) / (max_val - min_val)

                return norm_data, z_vals, mu, sigma, min_val, max_val

            if salida == "linear":
                norm_data = self.normalizar_mu_sigma(forw, mu, sigma)
                return norm_data, z_vals, mu, sigma
            


        elif data_mode == "desnorm":
            return forw, z_vals

        else:
            raise ValueError("data_mode debe ser 'norm' o 'desnorm'")
    


    def load_data_2d(self, file, data_mode):

        

        output = os.path.join("Camels_data", file)
        images, red = self.data0(output)

        images_ord, red_ord = self.reordenacion(images, red)

        delta = self.delta(images_ord)
        forw = forward(delta)
    
        #z_vals = self.normalizar_z(red)
        z_vals = self.factor_escala(red_ord)


        if data_mode == "global_norm_tanh":
            norm_data, max_desnorm, min_desnorm = self.normalizar_datos_tanh(forw)
            return norm_data, z_vals, max_desnorm, min_desnorm 

        elif data_mode == "redshift_norm":
            mu, sigma = self.compute_mu_sigma(forw) #Se lo doy por evoluciones porque compute ya lo reagrupa dentro, y salen por evoluciones, como los datos
            norm_data = self.normalizar_mu_sigma(forw, mu, sigma)
            return norm_data, z_vals, mu, sigma

        elif data_mode == "forw": #Este para el caso linear sin hacer la "norm" de mu y sigma
            forw = np.expand_dims(forw, -1)
            return forw, z_vals


        #Trrndría que añadir otro más en caso de querer hacer ambas cosas; hacer la norm de redshift y luego tanh

        else:
            raise ValueError("Elige bien el data_mode, 'global_norm' o 'redshift_norm'")
