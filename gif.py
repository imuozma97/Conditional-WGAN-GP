"""
gifs
"""

import os
import imageio

def gif(image_folder, output_name):

    image_files = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.png') or fname.endswith('.jpg')])
    images = [imageio.imread(f) for f in image_files]
    imageio.mimsave(os.path.join(image_folder, output_name), images, duration=400)