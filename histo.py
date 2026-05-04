"""
Archivo para generar las funciones de los histogramas
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from config import num_classes, N, num_cv

class Histogramas:
    def __init__(self, generated_images_folder, redshifts):
        self.generated_images_folder = generated_images_folder
        self.redshifts = redshifts

    def histograma(self, data1, data2, tipo, epoch,  i = None):
        """
        data1: datos generados
        data2: datos reales
        """
        values = data1.numpy().flatten() if hasattr(data1, "numpy") else data1.flatten()
        values2 = data2.numpy().flatten() if hasattr(data2, "numpy") else data2.flatten()

        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "Fake")
        plt.hist(values2, bins=50, color='purple', edgecolor='black', alpha=0.7, label = "Real")
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número de vóxeles")
        plt.title("Distribución en z = {}".format(self.redshifts[i]))

        plt.yscale('log')

        plt.grid(False)
        plt.legend()
        plt.ylim(1, 10**7)

        if tipo == "norm":
            plt.xlim(-1, 1)
            filename = f"histo_norm_{i:02d}.png"
            carpeta = f"histogramas_normalizados_{epoch}"
            if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
                os.makedirs(os.path.join(self.generated_images_folder, carpeta))

        if tipo == "desnorm":
            plt.xlim(0, 4500)
            filename = f"histo_desnorm_{i:02d}.png"
            carpeta = f"histogramas_desnormalizados_{epoch}"
            if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
                os.makedirs(os.path.join(self.generated_images_folder, carpeta))
        
        
        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()

    def all_histogramas(self, fake_agrupado, real_agrupado, tipo, epoch):
        for i in range(num_classes):
            self.histograma(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, i)