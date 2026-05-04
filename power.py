"""
Archivo que contiene la clase Power
"""
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

from config import boxsize, image_size, num_classes


class Power(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.precompute_k_and_bins()
        
        
    def precompute_k_and_bins(self):
        L = image_size

        PI = tf.constant(3.141592653589793, dtype=tf.float32)

        dx = boxsize / L

        freqs = tf.concat([tf.range(0, L//2, dtype = tf.float32), tf.range(-L//2, 0, dtype = tf.float32) ], axis = 0)
        freqs = freqs / (L*dx)*2.0*PI
        freqs = tf.signal.fftshift(freqs)

        kx, ky, kz = tf.meshgrid(freqs, freqs, freqs, indexing="ij")
        k_mag = tf.sqrt(kx**2 + ky**2 + kz**2)

        self.nbins = L // 2
        k_max = tf.reduce_max(k_mag)

        bin_edges = tf.linspace(0.0, k_max, self.nbins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        #SOLUCIÓN AL ERROR
        k_mag_flat = tf.reshape(k_mag, [-1])
        bin_indices_flat = tf.searchsorted(
            bin_edges, k_mag_flat, side="right"
        ) - 1

        bin_indices_flat = tf.clip_by_value(
            bin_indices_flat, 0, self.nbins - 1
        )

        bin_indices = tf.reshape(bin_indices_flat, k_mag.shape)

        # guardar constantes
        self.k_mag = tf.constant(k_mag)
        self.bin_indices = tf.constant(bin_indices)
        self.bin_centers = tf.constant(bin_centers)


           
    def compute_psd(self, image):
        """
        Computa PSD para una imagen
        image: (L, L, L)
        Retorna: (psd, bin_centers)
        """
        image = tf.cast(image, tf.float32)
        
        # FFT 3D
        fft3 = tf.signal.fft3d(tf.cast(image, tf.complex64))
        power = tf.math.abs(fft3)**2 / (boxsize**3)
        
        # centrar k=0
        power = tf.signal.fftshift(power, axes=(-3, -2, -1))
        
        # aplanar
        power_flat = tf.reshape(power, [-1])
        bin_idx_flat = tf.reshape(self.bin_indices, [-1])
        
        # media por bin radial
        psd = tf.math.unsorted_segment_mean(
            data=power_flat,
            segment_ids=bin_idx_flat,
            num_segments=self.nbins
        )
        
        return psd, self.bin_centers 
    
    
    
    def compute_all_psd(self, images):
        """
        Procesa todas las imágenes del batch de forma vectorizada
        images shape: (batch, 64, 64, 64, 1)
        """
        # Eliminar la dimensión de canales
        images = tf.squeeze(images, axis=-1)
        
        # Vectorizado: procesar todas las imágenes a la vez
        psd_results = tf.map_fn(
            lambda x: self.compute_psd(x)[0],
            images,
            fn_output_signature=tf.float32
        )
        
        return psd_results



    def compute_mean(self, psds):
        
        """
        Calcula el psd medio del conjunto de psd que se le pasen. También saca el psd_max el psd_min y el sigma de ese conjunto de psd
        """
        
        psd_mean_list = []  

        for psd in psds:
            psd_mean_list.append(psd)
        
        psd_mean_list = np.array(psd_mean_list)
        psd_tensor = tf.stack(psd_mean_list, axis=0)
        mean_psd = tf.reduce_mean(psd_tensor, axis=0)
        sigma = np.std(psd_tensor, axis = 0)
        
        
        psd_max = np.max(psd_mean_list, axis = 0)
        psd_min = np.min(psd_mean_list, axis = 0)
        
        return mean_psd, psd_max, psd_min, sigma
        
        
        

    def compute_all_mean(self, psds, group):
        
        """
        Calcula todas las medias de los psd. Por eso llama a compute_mean, para cacular la media, el psd_max, el psd_min y el sigma
        de cada conjunto del mismo redshift y los añade a un vector
        """
        
        psd_all_mean = []
        psd_max_mean = []
        psd_min_mean = []
        sigmas = []
        
        for i in range(num_classes):
            
            psd = self.compute_mean(psds[group*i : group + group*i])
            psd_all_mean.append(psd[0])
            psd_max_mean.append(psd[1])
            psd_min_mean.append(psd[2])
            sigmas.append(psd[3])
            
            
        psd_all_mean = np.array(psd_all_mean)
        psd_max_mean = np.array(psd_max_mean)
        psd_min_mean = np.array(psd_min_mean)
        sigmas = np.array(sigmas)
        
        return psd_all_mean, psd_max_mean, psd_min_mean, sigmas
    
    
    
    
    
    def compare_psd(self, k_values, mean_real, mean_fake, psd_max_real, psd_min_real, psd_max_fake, psd_min_fake, redshift, generated_images_folder, carpeta, tipo):
 
        for i in range(num_classes):
            plt.figure(figsize=(8, 5))

            plt.plot(k_values, mean_real[i], '-o', ms = 4, color = 'blue', label = "Real")
            plt.plot(k_values, mean_fake[i], '-o', ms = 4, color = 'red', label = "Fake")
        
            plt.fill_between(k_values, psd_min_real[i], psd_max_real[i], color='blue', alpha = 0.2, label = "max-min real")
            plt.fill_between(k_values, psd_min_fake[i], psd_max_fake[i], color='red', alpha = 0.2, label = "max-min fake")

            plt.yscale('log')
            plt.xlabel("$k$ [h/Mpc]", fontsize = 20)
            plt.ylabel("P(k)", fontsize = 20)

            plt.title("PSD vs. $k$ at z = {:.2f}".format(float(redshift[i])), fontsize = 24)
            plt.legend(fontsize = 14)
            if tipo == "norm":
                plt.ylim(10**-4, 10**5)
            elif tipo == "desnorm":
                plt.ylim(10**-2, 10**6)

            
            if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
                os.makedirs(os.path.join(generated_images_folder, carpeta))

            plt.savefig(os.path.join(generated_images_folder , carpeta, f"Compare_psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()
