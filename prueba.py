from preprocess_data import Dataset
from config import batch_size1, n_bar, num_cv
import numpy as np
from transforms import forward, backward
import h5py
import tensorflow as tf

datos = Dataset(batch_size1, n_bar)
#images, red = datos.data0('Camels_data/Data128.hdf5')
#print("MAx", np.max(images))
#print("MAx", np.max(backward(images)))

#back_im = backward(images)

file = "Camels_data/Datos64_nuevos.hdf5"
f = h5py.File(file, 'r')
maps = f['train_maps'][:]
red = np.array(f['train_labels'])[:]

print(red[0:34])