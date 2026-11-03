"""
Archivo que genera las cajas HR completas a partir de las cajas LR con el generador entrenado (superresolución).
El generador es totalmente convolucional: se recorre la caja LR por trozos (teselas) con margen y se pega solo
el centro de cada salida. La caja es periódica, así que las teselas del borde cogen celdas del otro lado.
"""

import tensorflow as tf
import os
import numpy as np
import h5py
import hdf5plugin

from Architectures_sr.generator_sr import Generator_sr
from transforms_sr import forward_bloques, backward


class Fake_images_sr(tf.keras.Model):

    def __init__(self, trained_models_folder, generated_images_folder, margen, centro_LR = 64, lote = 8):
        """
        margen: celdas LR que se añaden a cada lado de la tesela y luego se descartan (el mismo que en el entrenamiento).
        centro_LR: lado LR de la parte que se pega de cada tesela (tiene que dividir la malla LR: 512/64 = 8).
        lote: teselas que pasan juntas por el generador.
        """
        super().__init__()
        self.trained_models_folder = trained_models_folder
        self.generated_images_folder = generated_images_folder
        self.margen = margen
        self.centro_LR = centro_LR
        self.lote = lote


    def cargar_generador(self, name, filter1 = 64, filter2 = 32, filter3 = 16):
        """
        Crea el Generator_sr y carga los pesos guardados (name: p. ej. "best_psd_generator/epoch_00500").
        No se usa load_model porque el modelo guardado solo admite entradas de 64^3 (el tamaño del entrenamiento)
        y aquí las teselas son de centro_LR + 2*margen. Los filtros tienen que ser los del entrenamiento.
        """
        generator = Generator_sr(filter1 = filter1, filter2 = filter2, filter3 = filter3)
        generator([tf.zeros([1, 8, 8, 8, 1]), tf.zeros([1, 1])], training = False)   # crea los pesos
        generator.load_weights(os.path.join(self.trained_models_folder, name, "variables", "variables")).expect_partial()
        return generator


    def generar_caja(self, generator, caja_LR, z):
        """
        caja_LR: (g, g, g) ya transformada con forward.    z: factor de escala (el mismo que en el entrenamiento).
        Devuelve delta HR generada (2g, 2g, 2g), ya con backward.
        """
        g = caja_LR.shape[0]
        c, m = self.centro_LR, self.margen
        if g % c != 0:
            raise ValueError(f"centro_LR ({c}) tiene que dividir la malla LR ({g})")

        # Caja LR ampliada periódicamente con el margen: (g + 2m)^3
        caja_ext = np.pad(caja_LR, m, mode = "wrap")
        caja_HR = np.zeros((2*g, 2*g, 2*g), dtype = np.float32)

        posiciones = [(i, j, k) for i in range(0, g, c) for j in range(0, g, c) for k in range(0, g, c)]
        for p in range(0, len(posiciones), self.lote):
            trozo = posiciones[p:p + self.lote]
            teselas = np.stack([caja_ext[i:i + c + 2*m, j:j + c + 2*m, k:k + c + 2*m] for i, j, k in trozo])[..., np.newaxis]
            z_lote = np.full((len(trozo), 1), z, dtype = np.float32)

            salida = generator([teselas, z_lote], training = False)                  # (lote, 2(c+2m), ..., 1)
            salida = salida[:, 2*m:2*m + 2*c, 2*m:2*m + 2*c, 2*m:2*m + 2*c, 0]      # centro de cada tesela
            salida = (backward(salida) - 1).numpy()                                  # delta HR

            for (i, j, k), s in zip(trozo, salida):
                caja_HR[2*i:2*i + 2*c, 2*j:2*j + 2*c, 2*k:2*k + 2*c] = s

        return caja_HR


    def generar_simulacion(self, generator, carpeta_LR, indice, carpeta_salida, factor_escala):
        """
        Genera las 5 cajas HR de una simulación a partir de su archivo LR (<carpeta_LR>/<indice:04d>.hdf5)
        y las guarda en <carpeta_salida>/<indice:04d>.hdf5 con el mismo formato que las mallas:
        maps (5, 2g, 2g, 2g) con delta en float32 y redshift (5,).
        factor_escala: la función que convierte redshift en la condición (datos.factor_escala).
        Una simulación HR de 1024^3 ocupa ~21 GB en float32.
        """
        os.makedirs(carpeta_salida, exist_ok = True)
        f_LR = h5py.File(os.path.join(carpeta_LR, f"{indice:04d}.hdf5"), 'r')
        if not np.all(f_LR['completado'][:]):
            raise ValueError(f"Simulación {indice}: hay snapshots sin procesar")
        red = np.array(f_LR['redshift'][:], dtype = np.float32)
        g = f_LR['maps'].shape[1]
        n_bar = float(f_LR.attrs['N_particulas']) / g**3

        f_out = h5py.File(os.path.join(carpeta_salida, f"{indice:04d}.hdf5"), 'w')
        maps = f_out.create_dataset('maps', shape = (len(red), 2*g, 2*g, 2*g), dtype = np.float32,
                                    chunks = (1, 128, 128, 128))
        f_out.create_dataset('redshift', data = red)
        f_out.attrs['indice'] = indice

        for s in range(len(red)):
            delta_LR = (f_LR['maps'][s].astype(np.float32) - n_bar) / n_bar
            caja_LR = forward_bloques(delta_LR[np.newaxis])[0]
            z = factor_escala(red[s:s+1])[0]
            maps[s] = self.generar_caja(generator, caja_LR, z)
            print(f"Simulación {indice}, z = {red[s]:.2f} generada", flush = True)

        f_LR.close()
        f_out.close()