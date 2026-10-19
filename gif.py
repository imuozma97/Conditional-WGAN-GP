"""
gifs
"""

import os
import imageio

def gif(image_folder, output_name):

    image_files = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')])
    images = [imageio.imread(f) for f in image_files]
    imageio.mimsave(os.path.join(image_folder, output_name), images, duration=250)





gif("Results-64/baseline-images/compare_histo_claude_00683", "gif_histo_683.gif")
gif("Results-64/baseline-images/compare_psd_claude_percentil_00683", "gif_psd_683.gif")

gif("Results-64/psd_dyn-images/compare_histo_claude_00663", "gif_histo_663.gif")
gif("Results-64/psd_dyn-images/compare_psd_claude_percentil_00663", "gif_psd_663.gif")

gif("Results-128/baseline-images/compare_histo_claude_00439", "gif_histo_439.gif")
gif("Results-128/baseline-images/compare_psd_claude_percentil_00439", "gif_psd_439.gif")

gif("Results-128/psd_dyn-images/compare_histo_claude_00422", "gif_histo_422.gif")
gif("Results-128/psd_dyn-images/compare_psd_claude_percentil_00422", "gif_psd_422.gif")



