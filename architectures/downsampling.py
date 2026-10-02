from preprocess_data import Dataset
from config import batch_size1, buffer_size
import tensorflow as tf


class Downsample(tf.keras.Model):

    def __init__(self, n_bar_64, n_bar_128):
        super().__init__()
        self.n_bar_64 = n_bar_64 #Tiene que ser input porque no siempre es el mismo
        self.n_bar_128 = n_bar_128

        self.dataset = Dataset(batch_size1, buffer_size)

    def downsample_128(self, fake_delta_128):
        # 1- Recibe la delta de 128 falsa y lo pasa a número de partículas de 128 falso
        fake_npart_128 = self.dataset.deshacer_delta(fake_delta_128, self.n_bar_128)

        # 2- Calculo el cubo de número de partículas de 64 tras reducir el de 128 falso
        downsample = tf.keras.layers.AveragePooling3D(pool_size=2,strides=2, padding="valid")
        fake_npart_64 = downsample(fake_npart_128) * 8.0

        # 3- Calculo la delta de 64 falso, que será lo que compare con el 64 
        fake_delta_64 = self.dataset.delta(fake_npart_64, self.n_bar_64)

        return fake_delta_64