import os
import imageio
import h5py
import matplotlib.pyplot as plt
import numpy as np
from gif import gif
from preprocess_data import Dataset
from config import batch_size1, ncritic4, n_bar2, image_size2, num_cv, num_classes


datos= Dataset(batch_size1, n_bar2, buffer_size = 918)
n_part, red = datos.load_npart("Data3D-64.hdf5")
delta = datos.delta(n_part)
print("max delta", np.max(delta))





"""
for idx, img in enumerate(images):

    plt.figure(figsize=(5,5),  facecolor="black")
    plt.imshow(img)
    plt.title(f"$z = {redshifts[idx]:.2f}$", fontsize=14, color="white")
    plt.axis("off")

    path = os.path.join(image_folder, "cubo_redshift")
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, f"cubo_{idx:02d}.jpg"), dpi=300,
                bbox_inches='tight', format='jpg')
    plt.close()


gif(os.path.join(image_folder, "cubo_redshift"), "cubo_gif_kiara.gif")
"""