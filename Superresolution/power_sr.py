"""
Archivo que contiene la clase Power
"""
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os


class Power(tf.keras.Model):
    """
    binning="round": bins de anchura k_F centrados en n*k_F, n = 1..L/2-1 (31 bins para L=64).
    binning="floor": bins [n k_F, (n+1) k_F), como Pylians (solo para validar).
    Sin k = 0 y sin modos por encima de k_Ny.
    physical_units=False: |FFT|^2 / boxsize^3 (entrenamiento, pérdida, PSD_delta.npz).
    physical_units=True:  |FFT|^2 * boxsize^3 / L^6 -> (Mpc/h)^3 (solo figuras).
    deconvolve_ngp=True: divide cada modo por W_NGP^2 (solo figuras).
    """

    def __init__(self, image_size, boxsize, binning="round", nbins=None, physical_units=False, deconvolve_ngp=False):
        super().__init__()
        self.image_size = image_size
        self.boxsize = boxsize
        self.binning = binning
        self.physical_units = physical_units
        self.deconvolve_ngp = deconvolve_ngp
        self.nbins = image_size // 2 - 1 if nbins is None else nbins
        self.precompute_k_and_bins()


    def precompute_k_and_bins(self):
        L = self.image_size
        self.k_F = 2.0 * np.pi / self.boxsize
        self.k_Ny = np.pi * L / self.boxsize

        # Índices enteros de los modos en el orden de la FFT (ya no hace falta fftshift)
        n = np.fft.fftfreq(L, d=1.0 / L)
        nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")
        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)

        if self.binning == "round":
            b = np.rint(n_mag).astype(np.int64)
            k_centers = np.arange(1, self.nbins + 1) * self.k_F
        else:
            b = np.floor(n_mag).astype(np.int64)
            k_centers = (np.arange(1, self.nbins + 1) + 0.5) * self.k_F

        valid = (b >= 1) & (b <= self.nbins)
        # segmentos 0..nbins-1 = bins válidos; segmento nbins = descartados (k=0 y k>último bin)
        seg = np.where(valid, b - 1, self.nbins).astype(np.int32)

        n_modes = np.bincount(seg.ravel(), minlength=self.nbins + 1)[:self.nbins]
        k_sum = np.bincount(seg.ravel(), weights=(n_mag * self.k_F).ravel(), minlength=self.nbins + 1)[:self.nbins]

        W = np.sinc(nx / L) * np.sinc(ny / L) * np.sinc(nz / L)   # ventana NGP

        norm = self.boxsize**3 / float(L)**6 if self.physical_units else 1.0 / self.boxsize**3
        factor = np.full(n_mag.shape, norm, dtype=np.float64)
        if self.deconvolve_ngp:
            factor = factor / W**2

        self.segment_ids = tf.constant(seg.ravel(), dtype=tf.int32)
        self.mode_factor = tf.constant(factor.astype(np.float32))
        self.n_modes = n_modes
        self.k_centers = k_centers.astype(np.float32)
        self.k_eff = (k_sum / n_modes).astype(np.float32)
        self.bin_centers = tf.constant(self.k_centers)



    def compute_psd(self, image):
        image = tf.cast(image, tf.float32)
        fft3 = tf.signal.fft3d(tf.cast(image, tf.complex64))
        # Re^2 + Im^2 en vez de abs()**2: evita gradiente NaN si F = 0
        power = tf.math.square(tf.math.real(fft3)) + tf.math.square(tf.math.imag(fft3))
        power = power * self.mode_factor

        psd = tf.math.unsorted_segment_mean(
            data=tf.reshape(power, [-1]),
            segment_ids=self.segment_ids,
            num_segments=self.nbins + 1
        )[:self.nbins]

        return psd, self.bin_centers
    

    
    def compute_all_psd(self, images):
        """
        Procesa todas las imágenes del batch de forma vectorizada
        """
        images = tf.squeeze(images, axis=-1)
        
        # En lugar de la lambda, definimos la función aquí dentro
        def evaluar_psd(x):
            return self.compute_psd(x)[0]

        # Pasamos 'evaluar_psd' en lugar de la lambda
        psd_results = tf.map_fn(evaluar_psd, images, fn_output_signature=tf.float32)
        return psd_results



    