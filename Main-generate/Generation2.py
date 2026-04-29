"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas
"""
"""

import tensorflow as tf
import os
import numpy as np
from models import Fake_images
from preprocess_data import Dataset
from config import batch_size1, num_classes, image_size, latent_dim, num_cv, mass, boxsize, n_bar, N

#Esta parte es necesaria para poder tener z_vals. Voy a mirar de hacerlo por separado para no tener que hacer todo lo de los datos innecesariamente
#datos= Dataset(batch_size1)
#_, z_vals, _, _ = datos.load_data("norm")


#imagenes = Fake_images()
#gen_images = imagenes.generate_images(z_vals)
#imagenes.save_data("datos_gen_1201.npz", gen_images[0], gen_images[1]) #Aquí duda sobre el nombre del archivo, si yo quiero que sea genérico elegirlo desde otro archivo
