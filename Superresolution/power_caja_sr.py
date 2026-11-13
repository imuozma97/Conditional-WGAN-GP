"""
Archivo que contiene la clase Power_caja_sr: P(k) de las cajas completas (1024^3) para comparar
cada HR generada con SU HR real (misma simulación), al estilo de compare_psd_final de power.py.
No se usa la clase Power porque para 1024^3 sus tablas de modos ocuparían decenas de GB:
aquí la FFT se hace con scipy y los bins se acumulan plano a plano (mismos bins: anchura k_F, centrados en n*k_F).
"""
import os
import numpy as np
import h5py
import hdf5plugin
import scipy.fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Power_caja_sr:

    def __init__(self, boxsize):
        self.boxsize = boxsize          # lado de la caja completa en Mpc/h (Quijote: 1000)


    def leer_delta(self, archivo, s):
        """delta de un snapshot. Mallas (recuentos): n/n̄ - 1. Generados (float32): ya es delta."""
        f = h5py.File(archivo, 'r')
        caja = f['maps'][s]
        if np.issubdtype(caja.dtype, np.integer):
            n_bar = float(f.attrs['N_particulas']) / caja.size
            caja = caja.astype(np.float32) / n_bar - 1
        f.close()
        return caja.astype(np.float32, copy = False)


    def compute_psd(self, delta):
        """
        P(k) de una caja periódica en (Mpc/h)^3 (|FFT|^2 * boxsize^3 / L^6, como physical_units=True),
        bins n = 1 .. L/2. Devuelve k (medio de cada bin) y P.
        """
        L = delta.shape[0]
        F = scipy.fft.rfftn(delta, workers = -1)
        n = np.fft.fftfreq(L, d = 1.0 / L)
        nz = np.arange(F.shape[2])
        nyz2 = n[:, None]**2 + nz[None, :]**2
        # rfft guarda medio espacio: los planos 0 < kz < L/2 cuentan también por su conjugado (peso 2)
        peso = np.where((nz > 0) & (nz < L / 2), 2.0, 1.0)[None, :].repeat(L, 0).ravel()
        nbins = L // 2
        suma, modos, k_suma = np.zeros(nbins + 1), np.zeros(nbins + 1), np.zeros(nbins + 1)
        for i in range(L):
            nmod = np.sqrt(n[i]**2 + nyz2).ravel()
            b = np.rint(nmod).astype(np.int64)
            b[b > nbins] = 0                                  # bin 0 (k = 0 y k > k_Ny): se descarta
            suma += np.bincount(b, weights = peso * (np.abs(F[i])**2).ravel(), minlength = nbins + 1)
            modos += np.bincount(b, weights = peso, minlength = nbins + 1)
            k_suma += np.bincount(b, weights = peso * nmod, minlength = nbins + 1)
        k_F = 2 * np.pi / self.boxsize
        return (k_suma[1:] / modos[1:]) * k_F, suma[1:] / modos[1:] * self.boxsize**3 / float(L)**6


    def compute_all_psd(self, archivo):
        """P(k) de los 5 snapshots de un archivo (orden de z del archivo). Devuelve k, psd (5, nbins) y redshift."""
        with h5py.File(archivo, 'r') as f:
            red = f['redshift'][:]
        psds = []
        for s in range(len(red)):
            k, P = self.compute_psd(self.leer_delta(archivo, s))
            psds.append(P)
        return k, np.array(psds), red


    def espectro(self, F1, F2 = None):
        """
        Media por bins (los mismos que compute_psd) de Re(F1 F2*) * boxsize^3 / L^6. Sin F2: espectro de F1.
        F1, F2: rfftn de dos cajas del mismo tamaño.
        """
        L = F1.shape[0]
        n = np.fft.fftfreq(L, d = 1.0 / L)
        nz = np.arange(F1.shape[2])
        nyz2 = n[:, None]**2 + nz[None, :]**2
        peso = np.where((nz > 0) & (nz < L / 2), 2.0, 1.0)[None, :].repeat(L, 0).ravel()
        nbins = L // 2
        suma, modos = np.zeros(nbins + 1), np.zeros(nbins + 1)
        for i in range(L):
            b = np.rint(np.sqrt(n[i]**2 + nyz2)).astype(np.int64).ravel()
            b[b > nbins] = 0
            p = np.abs(F1[i])**2 if F2 is None else (F1[i] * np.conj(F2[i])).real
            suma += np.bincount(b, weights = peso * p.ravel(), minlength = nbins + 1)
            modos += np.bincount(b, weights = peso, minlength = nbins + 1)
        return suma[1:] / modos[1:] * self.boxsize**3 / float(L)**6



    def compute_all_cross(self, archivo_gen, archivo_real, archivo_LR):
        """
        Correlación cruzada r(k) = P_x / sqrt(P_1 P_2) de los 5 snapshots de una simulación:
          r_gen: HR generada frente a HR real.
          r_LR:  LR (ampliada a la malla HR repitiendo cada celda 2x2x2) frente a HR real, como referencia:
                 lo que se tendría sin superresolución.
        r = 1: misma estructura en el mismo sitio a esa escala; r = 0: sin relación.
        Devuelve k, r_gen (5, nbins), r_LR (5, nbins) y redshift. Memoria: ~20 GB para 1024^3.
        """
        with h5py.File(archivo_gen, 'r') as f:
            red = f['redshift'][:]
        r_gen, r_LR = [], []
        for s in range(len(red)):
            F_real = scipy.fft.rfftn(self.leer_delta(archivo_real, s), workers = -1)
            F_gen = scipy.fft.rfftn(self.leer_delta(archivo_gen, s), workers = -1)
            P_real = self.espectro(F_real)
            r_gen.append(self.espectro(F_gen, F_real) / np.sqrt(self.espectro(F_gen) * P_real))
            del F_gen
            F_LR = scipy.fft.rfftn(self.leer_delta(archivo_LR, s).repeat(2, 0).repeat(2, 1).repeat(2, 2), workers = -1)
            r_LR.append(self.espectro(F_LR, F_real) / np.sqrt(self.espectro(F_LR) * P_real))
            del F_LR, F_real
        L = 2 * h5py.File(archivo_LR, 'r')['maps'].shape[1]
        k = np.arange(1, L // 2 + 1) * 2 * np.pi / self.boxsize      # centros de los bins (n * k_F)
        return k, np.array(r_gen), np.array(r_LR), red



    def compare_cross_par(self, k_values, r_gen, r_LR, redshift, generated_images_folder, carpeta):
        """r(k) de una simulación, una figura por redshift: generada-real (rojo) y LR-real (gris, referencia)."""
        path = os.path.join(generated_images_folder, carpeta)
        os.makedirs(path, exist_ok = True)
        for i in range(len(redshift)):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_values, r_gen[i], color='red', linewidth=1.5, label="HR generada - HR real")
            ax.plot(k_values, r_LR[i], color='gray', linewidth=1.5, label="LR - HR real")
            ax.axhline(1, color='gray', linewidth=1, linestyle='--')
            ax.set_xscale('log')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(r"$r(k)$", fontsize=20)
            ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", fontsize=20)
            z = float(np.ravel(redshift)[i])
            z_str = f"{z:.1f}".rstrip("0").rstrip(".")
            ax.set_title(r"$z \sim " + z_str + r"$", fontsize=26)
            ax.tick_params(axis = 'both', labelsize = 16)
            ax.legend(fontsize=12)
            plt.savefig(os.path.join(path, f"cross_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.close(fig)



    def compare_psd_par(self, k_values, psd_fake, psd_real, redshift, generated_images_folder, carpeta):
        """
        Compara el P(k) de UNA caja HR generada con el de SU caja HR real (la misma simulación), por redshift.
        psd_fake, psd_real: (num_classes, nbins) de una simulación. Abajo, el residuo (P_fake - P_real) / P_real.
        Guarda <carpeta>/psd_00.png ... (una figura por redshift, en el orden del archivo).
        """
        eps = 1e-12
        for i in range(len(redshift)):

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, sharex=True)

            residuals = (psd_fake[i] - psd_real[i]) / (psd_real[i] + eps)

            ax1.plot(k_values, psd_real[i], '-o', ms=2, color='blue', label=r"$\mathrm{PSD}_{\mathrm{real}}$")
            ax1.plot(k_values, psd_fake[i], '-o', ms=2, color='red', label=r"$\mathrm{PSD}_{\mathrm{fake}}$")

            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylabel(r"$P(k)$ [$(h^{-1}\,\mathrm{Mpc})^3$]", fontsize=20)

            z = float(np.ravel(redshift)[i])
            z_str = f"{z:.1f}".rstrip("0").rstrip(".")
            ax1.set_title(r"$z \sim " + z_str + r"$", fontsize=26)
            ax1.tick_params(axis = 'y', labelsize = 18)
            ax1.legend(fontsize=14)

            ax2.plot(k_values, residuals, color='green', linewidth=1.5)
            ax2.axhline(0, color='gray', linewidth=1)
            ax2.set_ylabel(r'$\Delta P / P$', fontsize=20)
            ax2.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", fontsize=20)
            ax2.tick_params(axis = 'both', labelsize = 16)

            path = os.path.join(generated_images_folder, carpeta)
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(os.path.join(path, f"psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.close(fig)



    def residuos_todas(self, k_values, psd_fake, psd_real, redshift, indices, generated_images_folder, carpeta):
        """
        Residuos (P_fake - P_real) / P_real de cada par (una curva por simulación de test), una figura por redshift.
        psd_fake, psd_real: (simulaciones, num_classes, nbins). Muestra si el error se repite en todas las simulaciones.
        """
        eps = 1e-12
        path = os.path.join(generated_images_folder, carpeta)
        os.makedirs(path, exist_ok = True)
        for i in range(len(redshift)):
            fig, ax = plt.subplots(figsize=(8, 4))
            for n, ind in enumerate(indices):
                ax.plot(k_values, (psd_fake[n, i] - psd_real[n, i]) / (psd_real[n, i] + eps), linewidth=1, alpha=0.7, label=f"{ind}")
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_xscale('log')
            ax.set_ylabel(r'$\Delta P / P$', fontsize=20)
            ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", fontsize=20)
            z = float(np.ravel(redshift)[i])
            z_str = f"{z:.1f}".rstrip("0").rstrip(".")
            ax.set_title(r"$z \sim " + z_str + r"$", fontsize=26)
            ax.legend(fontsize=9, ncol=5)
            plt.savefig(os.path.join(path, f"residuos_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.close(fig)