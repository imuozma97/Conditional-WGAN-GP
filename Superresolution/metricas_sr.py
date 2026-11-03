"""
Métricas de la superresolución, par a par, sobre cajas completas (periódicas):
  - P(k) de la HR generada, la HR real y la LR (en (Mpc/h)^3, sin corregir ruido ni ventana NGP)
  - r(k): correlación cruzada generada-real y, como referencia, LR-real
  - T(k) = sqrt(P_gen / P_real)
  - histogramas de 1 + delta, fracción de celdas negativas y consistencia con la LR (bloques 2x2x2)
No usa la clase Power: para 1024^3 sus tablas ocuparían decenas de GB. Las FFT se hacen con scipy
(complex64) y los bins de k se acumulan plano a plano.
"""
import os
import numpy as np
import h5py
import hdf5plugin
import scipy.fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Metricas_sr:

    def __init__(self, boxsize = 1000.0, bins_hist = np.linspace(-3, 4, 141)):
        """
        boxsize: lado de la caja completa en Mpc/h (Quijote: 1000).
        bins_hist: bordes de los bins en log10(1 + delta). Los valores <= 10^-3 (celdas vacías) van al primer bin.
        """
        self.boxsize = boxsize
        self.bins_hist = bins_hist


    def leer_delta(self, archivo, s):
        """delta de un snapshot. Mallas (recuentos uint16): n/n̄ - 1. Generados (float32): ya es delta."""
        f = h5py.File(archivo, 'r')
        caja = f['maps'][s]
        if np.issubdtype(caja.dtype, np.integer):
            n_bar = float(f.attrs['N_particulas']) / caja.size
            caja = caja.astype(np.float32) / n_bar - 1
        f.close()
        return caja.astype(np.float32, copy = False)


    def fft(self, delta):
        return scipy.fft.rfftn(delta, workers = -1)        # float32 -> complex64


    def espectro(self, F1, F2 = None):
        """
        Media por bins de Re(F1 F2*) * L^3 / g^6. Sin F2: espectro de F1. Bins de anchura k_F centrados en n k_F,
        n = 1 .. g/2 (hasta k_Nyquist). Devuelve k medio de cada bin, el espectro y el número de modos.
        """
        g = F1.shape[0]
        n = np.fft.fftfreq(g, d = 1.0 / g)
        nz = np.arange(F1.shape[2])
        nyz2 = n[:, None]**2 + nz[None, :]**2
        # rfft guarda medio espacio: los planos 0 < kz < g/2 representan también a su conjugado (peso 2)
        peso = np.where((nz > 0) & (nz < g / 2), 2.0, 1.0)[None, :].repeat(g, 0).ravel()
        nbins = g // 2
        suma = np.zeros(nbins + 1)
        modos = np.zeros(nbins + 1)
        k_suma = np.zeros(nbins + 1)
        for i in range(g):
            nmod = np.sqrt(n[i]**2 + nyz2)
            b = np.rint(nmod).astype(np.int64).ravel()
            b[b > nbins] = 0                                   # el bin 0 (k = 0 y k > k_Ny) se descarta
            if F2 is None:
                p = np.abs(F1[i])**2
            else:
                p = (F1[i] * np.conj(F2[i])).real
            suma += np.bincount(b, weights = peso * p.ravel(), minlength = nbins + 1)
            modos += np.bincount(b, weights = peso, minlength = nbins + 1)
            k_suma += np.bincount(b, weights = peso * nmod.ravel(), minlength = nbins + 1)
        k_F = 2 * np.pi / self.boxsize
        norm = self.boxsize**3 / float(g)**6
        return (k_suma[1:] / modos[1:]) * k_F, suma[1:] / modos[1:] * norm, modos[1:]


    def histograma(self, delta):
        x = np.log10(np.clip(delta + 1, 1e-3, None))
        h, _ = np.histogram(x, bins = self.bins_hist, density = True)
        return h


    def consistencia(self, delta_HR, delta_LR):
        """
        Media de 1 + delta en bloques de 2x2x2 de la HR frente a 1 + delta de la LR, celda a celda:
        correlación de Pearson y diferencia rms. Con la HR real da el nivel que cabe esperar (más partículas).
        """
        g = delta_LR.shape[0]
        bloques = (delta_HR + 1).reshape(g, 2, g, 2, g, 2).mean(axis = (1, 3, 5))
        a = bloques.ravel() - 1
        b = delta_LR.ravel()
        return np.corrcoef(a, b)[0, 1], np.sqrt(np.mean((a - b)**2))


    def metricas_simulacion(self, archivo_LR, archivo_HR, archivo_gen):
        """
        Calcula todas las métricas de una simulación (sus 5 snapshots). Devuelve un diccionario de arrays,
        con el snapshot como primer índice. Memoria: ~20 GB para cajas de 1024^3.
        """
        with h5py.File(archivo_gen, 'r') as f:
            red = f['redshift'][:]
        res = {k: [] for k in ("P_real", "P_gen", "P_LR", "P_LR_up", "r_gen", "r_LR", "T",
                               "hist_real", "hist_gen", "hist_LR", "frac_neg", "media_gen",
                               "consist_real", "consist_gen")}

        for s in range(len(red)):
            delta_LR = self.leer_delta(archivo_LR, s)
            k_LR, P_LR, _ = self.espectro(self.fft(delta_LR))
            res["hist_LR"].append(self.histograma(delta_LR))

            d = self.leer_delta(archivo_HR, s)
            res["hist_real"].append(self.histograma(d))
            res["consist_real"].append(self.consistencia(d, delta_LR))
            F_real = self.fft(d)

            d = self.leer_delta(archivo_gen, s)
            res["hist_gen"].append(self.histograma(d))
            res["consist_gen"].append(self.consistencia(d, delta_LR))
            res["frac_neg"].append(np.mean(d < -1))           # celdas con 1 + delta < 0 (densidad negativa)
            res["media_gen"].append(np.mean(d))
            F_gen = self.fft(d)

            d = delta_LR.repeat(2, 0).repeat(2, 1).repeat(2, 2)  # LR en la malla HR (vecino más cercano)
            F_up = self.fft(d)
            del d

            k, P_real, modos = self.espectro(F_real)
            _, P_gen, _ = self.espectro(F_gen)
            _, P_up, _ = self.espectro(F_up)
            _, P_x_gen, _ = self.espectro(F_gen, F_real)
            _, P_x_LR, _ = self.espectro(F_up, F_real)
            del F_real, F_gen, F_up

            res["P_real"].append(P_real)
            res["P_gen"].append(P_gen)
            res["P_LR"].append(P_LR)
            res["P_LR_up"].append(P_up)
            res["r_gen"].append(P_x_gen / np.sqrt(P_gen * P_real))
            res["r_LR"].append(P_x_LR / np.sqrt(P_up * P_real))
            res["T"].append(np.sqrt(P_gen / P_real))
            print(f"   z = {red[s]:.2f}: r(k_max) gen = {res['r_gen'][-1][-1]:.3f}, "
                  f"T(k_max) = {res['T'][-1][-1]:.3f}, negativas = {res['frac_neg'][-1]*100:.2f} %", flush = True)

        res = {k: np.array(v) for k, v in res.items()}
        res.update(k = k, k_LR = k_LR, modos = modos, redshift = red, bins_hist = self.bins_hist)
        return res


    def guardar(self, archivo, res):
        os.makedirs(os.path.dirname(archivo), exist_ok = True)
        np.savez(archivo, **res)


    def figuras(self, archivos, carpeta):
        """
        Una figura por redshift con la media de todas las simulaciones de test (bandas: mínimo-máximo):
        P(k) con residuos, r(k), T(k) e histogramas.
        """
        os.makedirs(carpeta, exist_ok = True)
        R = [dict(np.load(a)) for a in archivos]
        k, k_LR, red, bins = R[0]["k"], R[0]["k_LR"], R[0]["redshift"], R[0]["bins_hist"]
        centros = 0.5 * (bins[1:] + bins[:-1])
        junta = lambda clave: np.stack([r[clave] for r in R])          # (simulaciones, snapshots, ...)

        for s in range(len(red)):
            fig, ax = plt.subplots(2, 2, figsize = (13, 9))

            for clave, kk, color, etiqueta in (("P_real", k, "blue", "HR real"), ("P_gen", k, "red", "HR generada"),
                                               ("P_LR", k_LR, "gray", "LR")):
                v = junta(clave)[:, s]
                ax[0, 0].loglog(kk, v.mean(0), color = color, label = etiqueta)
                ax[0, 0].fill_between(kk, v.min(0), v.max(0), color = color, alpha = 0.2)
            ax[0, 0].set(xlabel = r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", ylabel = r"$P(k)$ [$(h^{-1}\,\mathrm{Mpc})^3$]")
            ax[0, 0].legend()

            res_rel = junta("P_gen")[:, s] / junta("P_real")[:, s] - 1
            ax[0, 1].semilogx(k, res_rel.mean(0), color = "green")
            ax[0, 1].fill_between(k, res_rel.min(0), res_rel.max(0), color = "green", alpha = 0.2)
            ax[0, 1].axhline(0, color = "gray", lw = 1)
            ax[0, 1].set(xlabel = r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", ylabel = r"$P_{gen}/P_{real} - 1$")

            for clave, color, etiqueta in (("r_gen", "red", "generada - real"), ("r_LR", "gray", "LR - real")):
                v = junta(clave)[:, s]
                ax[1, 0].semilogx(k, v.mean(0), color = color, label = etiqueta)
                ax[1, 0].fill_between(k, v.min(0), v.max(0), color = color, alpha = 0.2)
            v = junta("T")[:, s]
            ax[1, 0].semilogx(k, v.mean(0), color = "red", ls = "--", label = r"$T(k)$ generada")
            ax[1, 0].axhline(1, color = "gray", lw = 1)
            ax[1, 0].set(xlabel = r"$k$ [$h\,\mathrm{Mpc}^{-1}$]", ylabel = r"$r(k)$, $T(k)$")
            ax[1, 0].legend()

            for clave, color, etiqueta in (("hist_real", "blue", "HR real"), ("hist_gen", "red", "HR generada"),
                                           ("hist_LR", "gray", "LR")):
                ax[1, 1].semilogy(centros, junta(clave)[:, s].mean(0), color = color, label = etiqueta)
            ax[1, 1].set(xlabel = r"$\log_{10}(1+\delta)$", ylabel = "PDF")
            ax[1, 1].legend()

            fig.suptitle(f"z = {red[s]:.2f}   ({len(R)} simulaciones de test)", fontsize = 16)
            fig.tight_layout()
            fig.savefig(os.path.join(carpeta, f"metricas_z{s}.png"), dpi = 130)
            plt.close(fig)

        print("Fracción de celdas negativas (media por z):", np.round(junta("frac_neg").mean(0) * 100, 3), "%")
        print("Media de delta generada (media por z):", np.round(junta("media_gen").mean(0), 4))
        print("Consistencia con la LR [corr, rms] real:", np.round(junta("consist_real").mean(0), 3).tolist())
        print("Consistencia con la LR [corr, rms] gen :", np.round(junta("consist_gen").mean(0), 3).tolist())