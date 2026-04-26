"""
Archivo que contiene la clase Power
"""
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

from config import boxsize, image_size


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
        image: (L, L, L) o (batch, L, L, L)
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
        Versión optimizada que procesa todas las imágenes a la vez
        """
        # images shape: (batch, 64, 64, 64, 1)
        batch_size = images.shape[0]
        
        # Calcular FFT para cada imagen del batch
        all_psds = []
        
        for i in range(batch_size):
            img = images[i, :, :, :, 0]  # (64, 64, 64)
            psd_val = self.compute_psd(img)[0]
            all_psds.append(psd_val)
            
        return tf.stack(all_psds)