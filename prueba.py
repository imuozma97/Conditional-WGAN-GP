from preprocess_data import Dataset
from config import batch_size1
import numpy as np
from transforms import forward, backward

datos = Dataset(batch_size1)
images, red = datos.data0('Camels_data/Data128.hdf5')
print("MAx", np.max(backward(images)))