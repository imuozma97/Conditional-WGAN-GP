"""
Este archivo va a ser el que genere los vtk de los datos reales.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from transforms import backward_2
from preprocess_data import Dataset
from config import batch_size1, image_size_64, num_cv, boxsize, n_bar_64

generated_images_folder = "Real-images"
trained_models_folder =  "Real-images"
datos= Dataset(batch_size1, buffer_size = 918)
N=100
imagenes = Fake_images(N = N, image_size = image_size_64, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

#DATOS REALES
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part, n_bar_64)
z_vals = datos.factor_escala(red)

imagenes.save_generated_vtk(delta, z_vals, output_folder=os.path.join(generated_images_folder, f"vtk_real_nuevo"), log_scale=True)
