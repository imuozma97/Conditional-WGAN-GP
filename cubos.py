"""
Archivo para generar los cubos falsos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

from config import num_classes

def cubo_part(data, redshift, carpeta_base, generated_images_folder):

    cubos_por_sim = num_classes
    num_sims = data.shape[0] // cubos_por_sim  # 27

    for sim in range(num_sims):

        # Crear carpeta Sim_i
        carpeta_sim = os.path.join(generated_images_folder, carpeta_base, f"Sim_{sim:02d}")
        os.makedirs(carpeta_sim, exist_ok=True)

        for j in range(cubos_por_sim):

            i = sim * cubos_por_sim + j  # índice global

            volume_fake = data[i]

            threshold = np.min(volume_fake)
            indices = np.argwhere(volume_fake > threshold)
            values = volume_fake[volume_fake > threshold]

            alpha = np.zeros_like(values, dtype=float)
            alpha[values < 2] = 0.3
            alpha[(values >= 2) & (values <= 6.5)] = 0.5
            alpha[values > 6.5] = 0.9

            norm = mcolors.Normalize(vmin=0, vmax=8)
            cmap = cm.plasma
            colors = cmap(norm(values))
            colors[:, 3] = alpha

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d', facecolor='black')

            ax.set_title(f'z = {redshift[i]}')
            ax.grid(False)

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("w")
            ax.yaxis.pane.set_edgecolor("w")
            ax.zaxis.pane.set_edgecolor("w")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            sc = ax.scatter(
                indices[:, 0], indices[:, 1], indices[:, 2],
                c=colors, marker='o', s=0.3
            )

            m = cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array(volume_fake)
            cbar = plt.colorbar(m, ax=ax, label='Valor del voxel')
            cbar.solids.set_edgecolor("face")

            # Guardar dentro de su simulación
            plt.savefig(
                os.path.join(carpeta_sim, f"z_{j:02d}.png"),
                bbox_inches='tight',
                format='png'
            )
            plt.close()