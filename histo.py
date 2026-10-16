"""
Archivo para generar las funciones de los histogramas
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from config import num_classes, num_cv
from scipy.interpolate import make_interp_spline

class Histogramas:
    def __init__(self, generated_images_folder, redshifts):
        self.generated_images_folder = generated_images_folder
        self.redshifts = redshifts

    def histograma(self, data1, data2, tipo, epoch,  i = None):
        """
        data1: datos generados
        data2: datos reales
        """
        print("fake shape: ", data1.shape)
        print("real shape: ", data2.shape)

        values = data1.numpy().flatten() if hasattr(data1, "numpy") else data1.flatten()
        values2 = data2.numpy().flatten() if hasattr(data2, "numpy") else data2.flatten()

        
        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "Fake")
        plt.hist(values2, bins=50, color='purple', edgecolor='black', alpha=0.7, label = "Real")
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número de vóxeles")
        plt.title("Distribución en z = {}".format(self.redshifts[i]))

        plt.yscale('log')

        plt.grid(False)
        plt.legend()
        plt.ylim(1, 10**7)

        filename = f"histo_{i:02d}.png"
        carpeta = f"histogramas_prueba_{epoch}"
        if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
            os.makedirs(os.path.join(self.generated_images_folder, carpeta))
        
        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()



    def all_histogramas(self, N, fake_agrupado, real_agrupado, tipo, epoch):
        for i in range(num_classes):
            self.histograma(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, i)



    def histo_fake(self, fake_data):

        values = fake_data.numpy().flatten() if hasattr(fake_data, "numpy") else fake_data.flatten()
        print("Values shape: ", values.shape)
        
        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "Fake")
        
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número de vóxeles")
       # plt.title("Distribución en z = {}".format(self.redshifts[i]))

        plt.yscale('log')

        plt.grid(False)
        plt.legend()
        plt.ylim(0.1, 10**7)

        
        #filename = f"histo_norm_{i:02d}.png"
        carpeta = f"histogramas_prueba"
        if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
            os.makedirs(os.path.join(self.generated_images_folder, carpeta))

        filepath = os.path.join(self.generated_images_folder, carpeta)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')



    def histo_individual(self, fake_data, real_data, i = None):

        print(fake_data.shape)
        print(real_data.shape)

        values = fake_data.numpy().flatten() if hasattr(fake_data, "numpy") else fake_data.flatten()
        values2 = real_data.numpy().flatten() if hasattr(real_data, "numpy") else real_data.flatten()
        print("Values shape: ", values.shape)
        
        
        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label = "Fake")
        plt.hist(values2, bins=50, color='purple', edgecolor='black', alpha=0.7, label = "Real")
        
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número de vóxeles")
        # plt.title("Distribución en z = {}".format(self.redshifts[i]))

        plt.yscale('log')

        plt.grid(False)
        plt.legend()
        plt.ylim(1, 10**7)

        
            #filename = f"histo_norm_{i:02d}.png"
        carpeta = f"histogramas_prueba_{i}"
        if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
            os.makedirs(os.path.join(self.generated_images_folder, carpeta))

        filepath = os.path.join(self.generated_images_folder, carpeta)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')



    def histograma_medio(self, data1, data2, tipo, epoch, i=None):

        # Mismos bins para fake y real
        all_values = np.concatenate([
            data1.flatten(),
            data2.flatten()
        ])
        bins = np.linspace(all_values.min(), all_values.max(), 51)

        # Histograma medio fake
        hist_fake = []
        for sample in data1:
            h, _ = np.histogram(sample.flatten(), bins=bins)
            hist_fake.append(h)

        hist_fake = np.mean(hist_fake, axis=0)

        # Histograma medio real
        hist_real = []
        for sample in data2:
            h, _ = np.histogram(sample.flatten(), bins=bins)
            hist_real.append(h)

        hist_real = np.mean(hist_real, axis=0)

        # Centros de los bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = np.diff(bins)

        plt.figure(figsize=(6,4))
        plt.bar(centers, hist_fake, width=width,
                alpha=0.4, color='blue', edgecolor='black', linewidth=0.8, label='Fake')

        plt.bar(centers, hist_real, width=width,
                alpha=0.4, color='purple', edgecolor='black', linewidth=0.8, label='Real')

        plt.yscale('log')
        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número medio de vóxeles")
        plt.ylim(0.1, 10**6)
        plt.title(f"Distribución en z = {self.redshifts[i]}")
        plt.legend()

        filename = f"histo_{i:02d}.png"
        carpeta = f"histogramas_prueba_{epoch}"
        if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
            os.makedirs(os.path.join(self.generated_images_folder, carpeta))
        
        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        #plt.show()
        plt.close()

    

    def all_histogramas_medios(self, N, fake_agrupado, real_agrupado, tipo, epoch):
        for i in range(num_classes):
            self.histograma_medio(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, i)



    def histograma_medio_residuos_p90(self, data1, data2, tipo, epoch, redshift, carpeta, i=None):


        all_values = np.concatenate([data1.flatten(), data2.flatten()])
        bins = np.linspace(all_values.min(), all_values.max(), 51)

        # --- Histograma de referencia (real) ---
        hist_real_samples = []
        for sample in data2:
            h, _ = np.histogram(sample.flatten(), bins=bins)
            hist_real_samples.append(h)
        hist_real_samples = np.array(hist_real_samples)
        hist_real_mean = np.mean(hist_real_samples, axis=0)

        # --- Histograma fake por muestra ---
        hist_fake_samples = []
        for sample in data1:
            h, _ = np.histogram(sample.flatten(), bins=bins)
            hist_fake_samples.append(h)
        hist_fake_samples = np.array(hist_fake_samples)

        # --- Selección de los 90 más cercanos a la media real ---
        distances = np.linalg.norm(hist_fake_samples - hist_real_mean, axis=1)

        idx_sorted = np.argsort(distances)
        idx_selected = idx_sorted[:90]

        hist_fake_selected = hist_fake_samples[idx_selected]
        hist_fake = np.mean(hist_fake_selected, axis=0)

        # --- Histograma real medio ---
        hist_real = np.mean(hist_real_samples, axis=0)

        # Centros
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = np.diff(bins)

        # Residuo relativo
        residual = np.zeros_like(hist_real, dtype=float)
        mask = hist_real > 0
        residual[mask] = (hist_fake[mask] - hist_real[mask]) / hist_real[mask]

        error_res = np.mean(np.abs(residual[mask]))
        print("Residual res: ", error_res)

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.03}, sharex=True)

        ax1.bar(centers, hist_fake, width=width, alpha=0.4, color='red', edgecolor='black', linewidth=0.8, label='Fake')
        ax1.bar(centers, hist_real, width=width, alpha=0.4, color='blue', edgecolor='black', linewidth=0.8, label='Real')

        ax1.set_yscale('log')
        ax1.set_ylim(0.1, 1e6)
        ax1.set_ylabel("N", fontsize=20)
        z = float(redshift[i])
        z_str = f"{z:.1f}".rstrip("0").rstrip(".")
        ax1.set_title(r"$z \sim " + z_str + r"$", fontsize=26)
        ax1.tick_params(axis = 'y', labelsize = 16)

        if i == 0:
            ax1.legend(fontsize=17)

        ax2.axhline(0, color='gray', linewidth=1)
        ax2.plot(centers, residual, color='green', markersize=3, linewidth=1.5)

        ax2.set_ylabel(r'$\Delta N / N$', fontsize=20)
        ax2.set_xlabel("$\delta$", fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis = 'both', labelsize = 16)

        filename = f"histo_{i:02d}.png"
        os.makedirs(os.path.join(self.generated_images_folder, carpeta), exist_ok=True)

        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()



    def all_histogramas_medio_residuos_p90(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, carpeta):
        for i in range(num_classes):
            self.histograma_medio_residuos_p90(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch,redshift, carpeta=f"histogramas_medio_residuos_p90_{epoch}", i=i)



    def histograma_final(self, data1, data2, tipo, epoch, redshift, carpeta, i = None,
                                      n_keep = 90, nb = 50, dequant = True, min_count = 10, seed = 0):
        """
        Histograma medio del número de celdas frente a 1+delta: generados (data1) frente a reales (data2).
          - Bins logarítmicos de igual anchura en log10(1+delta), fijados SOLO con los reales
            (iguales para todos los modelos), desde la celda real no vacía de menor densidad hasta la de mayor.
          - dequant: a cada celda real no vacía se le suma un ruido uniforme de +-0.5 partículas, para que
            los reales (conteos enteros / n_bar) sean comparables con los generados (continuos). Sin esto
            aparecen dientes de sierra artificiales en el histograma y en el residuo.
          - n_keep: media de los n_keep generados más cercanos a la media real (distancia en log); None = todos.
          - Residuo solo en bins con >= min_count celdas por cubo real; banda gris = +-1 sigma esperada
            solo por tener num_cv reales y n_keep generados.
        """
        n_bar = 256**3 / data2[0].size                          # partículas por celda: 64 para 64^3, 8 para 128^3
        rng = np.random.default_rng(seed)

        real = 1 + np.asarray(data2, dtype = np.float64).reshape(len(data2), -1)
        fake = 1 + np.asarray(data1, dtype = np.float64).reshape(len(data1), -1)
        if dequant:
            nz = real > 0
            real[nz] += rng.uniform(-0.5, 0.5, nz.sum()) / n_bar

        # Bins: solo con los reales, desde la celda real no vacía de menor densidad hasta la de mayor
        lo, hi = max(0.5 / n_bar, real[real > 0].min()), real.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), nb + 1)

        hist_real_samples = np.array([np.histogram(s, bins = bins)[0] for s in real], dtype = float)
        hist_fake_samples = np.array([np.histogram(s, bins = bins)[0] for s in fake], dtype = float)
        hist_real = hist_real_samples.mean(axis = 0)

        # Fuera del rango de los bins (se informan aparte)
        frac = lambda x, cond: np.mean(cond(x))
        vacias_real = frac(real, lambda x: x < lo)
        vacias_fake = frac(fake, lambda x: (x < lo) & (x >= 0))
        neg_fake = frac(fake, lambda x: x < 0)
        sobre_fake = frac(fake, lambda x: x > hi)

        # Selección de los n_keep más cercanos a la media real (distancia en log, como en el P(k))
        if n_keep is not None:
            distances = np.linalg.norm(np.log10(hist_fake_samples + 1) - np.log10(hist_real + 1), axis = 1)
            hist_fake_samples = hist_fake_samples[np.argsort(distances)[:min(n_keep, len(hist_fake_samples))]]
        hist_fake = hist_fake_samples.mean(axis = 0)

        # Residuo relativo y su nivel de ruido
        mask = hist_real >= min_count
        residual = np.full_like(hist_real, np.nan)
        residual[mask] = (hist_fake[mask] - hist_real[mask]) / hist_real[mask]
        ruido = np.full_like(hist_real, np.nan)
        ruido[mask] = np.sqrt(hist_real_samples[:, mask].var(axis = 0, ddof = 1) / len(hist_real_samples)
                              + hist_fake_samples[:, mask].var(axis = 0, ddof = 1) / len(hist_fake_samples)) / hist_real[mask]

        error_res = np.mean(np.abs(residual[mask]))
        floor_res = np.mean(np.sqrt(2 / np.pi) * ruido[mask])
        print(f"Residual res: {error_res:.4f}  (ruido esperado {floor_res:.4f})")

        # --- Plot ---
        centers = np.sqrt(bins[:-1] * bins[1:])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 5), gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.03}, sharex = True)

        ax1.bar(bins[:-1], hist_fake, width = np.diff(bins), align = 'edge', alpha = 0.4, color = 'red', edgecolor = 'black', linewidth = 0.8, label = 'Fake')

        ax1.bar(bins[:-1], hist_real, width = np.diff(bins), align = 'edge', alpha = 0.4, color = 'blue', edgecolor = 'black', linewidth = 0.8, label = 'Real')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim(0.1, 1e6 if data2[0].size <= 64**3 else 1e7)
        ax1.set_ylabel("N", fontsize = 20)
        z = float(np.ravel(redshift)[i])
        z_str = f"{z:.1f}".rstrip("0").rstrip(".")
        ax1.set_title(r"$z \sim " + z_str + r"$", fontsize = 26)
        ax1.tick_params(axis = 'y', labelsize = 16)
        txt = (f"below range (empty): real {vacias_real:.1e}, fake {vacias_fake:.1e}\n"
               f"fake $1+\\delta<0$: {neg_fake:.1e};  fake $>$ real max: {sobre_fake:.1e}")
        ax1.text(0.02, 0.97, txt, transform = ax1.transAxes, fontsize = 9, va = 'top')

        if i == 0:
            ax1.legend(fontsize = 17)

        ax2.fill_between(centers, -ruido, ruido, color = 'gray', alpha = 0.3, lw = 0)
        ax2.axhline(0, color = 'gray', linewidth = 1)
        ax2.plot(centers, residual, color = 'green', marker = 'o', markersize = 3, linewidth = 1.5)

        ax2.set_ylabel(r'$\Delta N / N$', fontsize = 20)
        ax2.set_xlabel(r"$1+\delta$", fontsize = 20)
        ax2.grid(True, alpha = 0.3)
        ax2.tick_params(axis = 'both', labelsize = 16)

        filename = f"histo_{i:02d}.png"
        os.makedirs(os.path.join(self.generated_images_folder, carpeta), exist_ok = True)

        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')
        plt.close()
        return {"error": error_res, "ruido": floor_res, "vacias_real": vacias_real, "vacias_fake": vacias_fake,
                "neg_fake": neg_fake, "sobre_fake": sobre_fake, "error_mean":np.mean(np.abs(error_res))}


    def all_histogramas_final(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, carpeta):
        for i in range(num_classes):
            self.histograma_final(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch,redshift, carpeta, i=i)



    def histograma_curva(self, data1, data2, tipo, epoch, redshift, carpeta, i = None,
                          n_keep = 90, nb = 50, dequant = True, min_count = 10, p_lo = 5, p_hi = 95, seed = 0, guardar_figuras = True):
        """
        Número de celdas por bin frente a 1+delta, en curvas: generados (data1) frente a reales (data2).
          - Bins logarítmicos de igual anchura en log10(1+delta), fijados SOLO con los reales.
          - dequant: ruido uniforme de +-0.5 partículas en las celdas reales no vacías (conteos enteros
            frente a generados continuos).
          - Curvas: media real y media de los n_keep generados más cercanos a la media real (None = todos).
            Bandas: percentiles p_lo-p_hi entre cubos, con TODOS los cubos.
          - Panel inferior: residuo relativo de la media, con la banda de +-1 sigma esperada.
        Errores (en bins con >= min_count celdas por cubo real de media):
          - err_media: |N_gen - N_real| / N_real medio (con los n_keep), y su nivel de ruido.
          - err_disp: |sigma_gen / sigma_real - 1| medio (con TODOS los generados), su nivel de ruido y su sesgo.
        """
        n_bar = 256**3 / data2[0].size                          # partículas por celda: 64 para 64^3, 8 para 128^3
        rng = np.random.default_rng(seed)

        real = 1 + np.asarray(data2, dtype = np.float64).reshape(len(data2), -1)
        fake = 1 + np.asarray(data1, dtype = np.float64).reshape(len(data1), -1)
        if dequant:
            nz = real > 0
            real[nz] += rng.uniform(-0.5, 0.5, nz.sum()) / n_bar

        # Bins: solo con los reales
        lo, hi = max(0.5 / n_bar, real[real > 0].min()), real.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
        centers = np.sqrt(bins[:-1] * bins[1:])

        h_r = np.array([np.histogram(s, bins = bins)[0] for s in real], dtype = float)
        h_f_todos = np.array([np.histogram(s, bins = bins)[0] for s in fake], dtype = float)
        m_r = h_r.mean(axis = 0)

        # Fuera del rango de los bins (se informan aparte)
        vacias_real = np.mean(real < lo)
        vacias_fake = np.mean((fake < lo) & (fake >= 0))
        neg_fake = np.mean(fake < 0)
        sobre_fake = np.mean(fake > hi)

        # Media de los generados: n_keep más cercanos a la media real (distancia en log)
        h_f = h_f_todos
        if n_keep is not None:
            d = np.linalg.norm(np.log10(h_f_todos + 1) - np.log10(m_r + 1), axis = 1)
            h_f = h_f_todos[np.argsort(d)[:min(n_keep, len(h_f_todos))]]
        m_f = h_f.mean(axis = 0)

        # Error de la media y su nivel de ruido
        ok = m_r >= min_count
        res = np.full_like(m_r, np.nan)
        res[ok] = (m_f[ok] - m_r[ok]) / m_r[ok]
        ruido = np.full_like(m_r, np.nan)
        ruido[ok] = np.sqrt(h_r[:, ok].var(axis = 0, ddof = 1) / len(h_r) + h_f[:, ok].var(axis = 0, ddof = 1) / len(h_f)) / m_r[ok]
        err_media = np.mean(np.abs(res[ok]))
        floor_media = np.mean(np.sqrt(2 / np.pi) * ruido[ok])

        # Error de la dispersión (todos los generados) y su nivel de ruido
        s_r = h_r.std(axis = 0, ddof = 1)
        s_f = h_f_todos.std(axis = 0, ddof = 1)
        ok_s = ok & (s_r > 0)
        rs = s_f[ok_s] / s_r[ok_s] - 1
        err_disp = np.mean(np.abs(rs))
        sesgo_disp = np.mean(rs)
        floor_disp = np.sqrt(2 / np.pi) * np.sqrt(1 / (2 * (len(h_r) - 1)) + 1 / (2 * (len(h_f_todos) - 1)))

        out = {"err_media": err_media, "floor_media": floor_media, "err_disp": err_disp, "floor_disp": floor_disp,
               "sesgo_disp": sesgo_disp, "vacias_real": vacias_real, "vacias_gen": vacias_fake,
               "neg_gen": neg_fake, "sobre_max_gen": sobre_fake}
        if not guardar_figuras:
            return out

        # --- Bandas de dispersión entre cubos (todos los cubos)
        lo_r, hi_r = h_r.min(axis = 0), h_r.max(axis = 0)                                  # todos los reales
        lo_f, hi_f = np.percentile(h_f_todos, [p_lo, p_hi], axis = 0)

        # --- Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 5.5), sharex = True,
                                       gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.05})

        ax1.fill_between(centers, lo_r, hi_r, color = 'blue', alpha = 0.2, lw = 0, label=r"$\mathrm{Range}_{\mathrm{real}}$")
        ax1.fill_between(centers, lo_f, hi_f, color = 'red', alpha = 0.2, lw = 0, label = r"$P_{%d}$–$P_{%d}$ fake" % (p_lo, p_hi))
        ax1.plot(centers, m_r, '-', color = 'blue', lw = 1.8, label = "Real (mean)")
        ax1.plot(centers, m_f, '--', color = 'red', lw = 1.8, label = "Fake (mean)")
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_ylim(0.1, 1e6 if data2[0].size <= 64**3 else 1e7)
        ax1.set_ylabel("N", fontsize = 20)
        z = float(np.ravel(redshift)[i])
        z_str = f"{z:.1f}".rstrip("0").rstrip(".")
        ax1.set_title(r"$z \sim " + z_str + r"$", fontsize = 26)
        ax1.tick_params(axis = 'y', labelsize = 16)
    
        if i == 0:
            ax1.legend(fontsize = 12, loc = 'upper right')

        ax2.fill_between(centers, -ruido, ruido, color = 'gray', alpha = 0.3, lw = 0)
        ax2.axhline(0, color = 'gray', linewidth = 1)
        ax2.plot(centers, res, '-', color = 'green', lw = 1.5)
        ax2.set_ylabel(r'$\Delta N / N$', fontsize = 20)
        ax2.set_xlabel(r"$1+\delta$", fontsize = 20)
        ax2.grid(True, alpha = 0.3)
        ax2.tick_params(axis = 'both', labelsize = 16)

        os.makedirs(os.path.join(self.generated_images_folder, carpeta), exist_ok = True)
        plt.savefig(os.path.join(self.generated_images_folder, carpeta, f"histo_{i:02d}.png"), dpi = 300, bbox_inches = 'tight')
        plt.close(fig)

        return out


    def all_histogramas_curva(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, carpeta, n_keep = 90, guardar_figuras = True):
        """Figuras (opcional) y errores de los histogramas en todos los redshifts. Los errores solo se imprimen."""
        n_z = len(fake_agrupado) // N           # número de redshifts de los datos que se pasan
        res = [self.histograma_curva(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i],
                                      tipo, epoch, redshift, carpeta = carpeta, i = i, n_keep = n_keep,
                                      guardar_figuras = guardar_figuras) for i in range(n_z)]
        z = np.ravel(redshift)[:n_z].astype(float)

        print(f"\n===== Errores de los histogramas ({N} generados, {num_cv} reales por redshift) =====")
        print("    z   err_media     ruido   err_disp     ruido     sesgo")
        for zi, r in zip(z, res):
            print(f"{zi:5.2f} {r['err_media']:10.4f} {r['floor_media']:9.4f} {r['err_disp']:10.4f} {r['floor_disp']:9.4f} {r['sesgo_disp']:+9.3f}")

        g = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        print("\nMedia sobre los redshifts:")
        print(f"  Error de la media:      {g['err_media']:.4f}   (ruido esperado {g['floor_media']:.4f})")
        print(f"  Error de la dispersión: {g['err_disp']:.4f}   (ruido esperado {g['floor_disp']:.4f}), sesgo {g['sesgo_disp']:+.3f}")
        print(f"  Fuera de rango (vacías): real {g['vacias_real']:.2e}, generados {g['vacias_gen']:.2e}")
        print(f"  Generados con 1+delta < 0: {g['neg_gen']:.2e}; por encima del máximo real: {g['sobre_max_gen']:.2e}")
        return res


