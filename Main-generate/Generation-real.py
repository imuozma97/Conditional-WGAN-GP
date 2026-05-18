"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from generate import Fake_images
from transforms import backward
from preprocess_data import Dataset
from power import Power
from config import batch_size1, num_classes, image_size, latent_dim, num_cv, mass, boxsize, n_bar
from histo import Histogramas
from cubos import cubo_part
from gif import gif
from psd_utils import lambda_psd_schedule

generated_images_folder = "Real-images"
datos= Dataset(batch_size1)
power = Power()
N=100
imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 

#DATOS REALES
#mu y sigma están ordenados por evoluciones
#_, red = datos.data0('Camels_data/Data3D-64.hdf5') y uso red en el compare_psd
norm_data, z_vals, mu, sigma = datos.load_data("redshift_norm")
desnorm_data = datos.desnormalizar_mu_sigma(norm_data, mu, sigma)

norm_data_agrupados, mu_agrupado, sigma_agrupado = datos.reordenacion(num_cv, norm_data, mu, sigma)
desnorm_data_agrupados = datos.desnormalizar_mu_sigma(norm_data_agrupados, mu_agrupado, sigma_agrupado)
desnorm_data_agrupados = backward(desnorm_data_agrupados)

imagenes.save_generated_vtk(desnorm_data, z_vals, output_folder=os.path.join(trained_models_folder, f"vtk_real"), log_scale=True)
