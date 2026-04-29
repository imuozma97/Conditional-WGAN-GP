"""
Archivo que genera los datos falsos y los guarda en un archivo tipo datos_gen_epoch dentro de la carpeta X-models
"""

import tensorflow as tf
import os
import numpy as np
from preprocess_data import Dataset
from config import num_classes, image_size, latent_dim, num_cv, mass, boxsize


class Fake_images(tf.keras.Model):
    
    def __init__(self, N, trained_models_folder, generated_images_folder):
        super().__init__()
        self.trained_models_folder =  trained_models_folder
        self.generated_images_folder = generated_images_folder
        self.N = N
    
    
    def generate_images(self, z_values, name): 
        """
        Genera imágenes cargando el modelo guardado.
        """
        model_path = os.path.join(self.trained_models_folder, name)
        generator = tf.keras.models.load_model(model_path, compile=False)

        generated_images = []
        redshift = []
        
        j = 1
        while j < self.N + 1:
            print('Evolución: ', j)
            
            noise = tf.random.normal([1, latent_dim])
            
            for i in range(num_classes):
                
                generated_data = generator([noise, np.expand_dims(z_values[i], 0)], training=False)
                
                generated_images.append(generated_data.numpy())
                redshift.append(z_values[i])
                
            j += 1
                    
        generated_images = np.array(generated_images).reshape(self.N*num_classes, image_size, image_size, image_size)
        redshift = np.array(redshift)

        return generated_images, redshift
    
    
    
    
    def density(self, data, red): #Esta es para sacar la densidad si se le dan los datos agrupados pero juntos, para que saque la densidad y separe
        
        density = data*mass/(boxsize/image_size)
        
        return density, red
  
    

    def density_mean(self, data):
        
        density_mean = []
        for i in range(len(data)):
        
            density_mean.append(np.mean(data[i]))
            
        density_mean = np.array(density_mean)
            
        return density_mean

    
    
    
    def save_data(self, name_output, data, labels):
        if not os.path.exists(name_output):
            os.makedirs(name_output)
        filepath = os.path.join(self.trained_models_folder, name_output)
        np.savez_compressed(filepath, data = data, labels = labels)
        

    
    def load_data(self, file):
        
        filepath = os.path.join(self.trained_models_folder, file)
        loaded = np.load(filepath)
        datos_fake = loaded["data"]  #Este sería el antiguo cubos generados
        labels_fake = loaded["labels"]
        
        return datos_fake, labels_fake