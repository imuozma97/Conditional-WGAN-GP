"""
Archivo para generar los cubos falsos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

from config import num_classes

def cubo_part(data, redshift, generated_images_folder):
    
    
    for i in range(num_classes):
    
        volume_fake = data[i]
        threshold = np.min(volume_fake)
        indices = np.argwhere(volume_fake > threshold)  # Coordenadas donde se supera el umbral
        values = volume_fake[volume_fake> threshold]        # Valores correspondientes

        alpha = np.zeros_like(values, dtype=float)

        alpha[values < 2] = 0.3
        alpha[(values >= 2) & (values <= 6.5)] = 0.5
        alpha[values > 6.5] = 0.9
        
        norm = mcolors.Normalize(vmin = 0, vmax = 8)

        cmap = cm.plasma
        colors = cmap(norm(values))

        # Sobrescribir canal alpha
        colors[:, 3] = alpha

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        ax.set_title('z = {}'.format(redshift[i]))
        ax.grid(False)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")  
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")


        #for spine in ax.spines.values():
            #spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")



        # Representar puntos
        sc = ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c = colors, marker = 'o', s = 0.3)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array(volume_fake)  # para que la colorbar abarque todo el volumen
        cbar = plt.colorbar(m, ax=ax, label='Valor del voxel')
        cbar.solids.set_edgecolor("face")

        carpeta = "Cubos"
        if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
            os.makedirs(os.path.join(generated_images_folder, carpeta))
        plt.savefig(os.path.join(generated_images_folder , carpeta, f"Sim_{i:02d}.png"), bbox_inches='tight', format='png')
        plt.close()