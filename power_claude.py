"""
Archivo que contiene la clase Power
"""
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

from config import boxsize, num_classes, num_cv


class Power(tf.keras.Model):
    """
    binning="round": bins de anchura k_F centrados en n*k_F, n = 1..L/2-1 (31 bins para L=64).
    binning="floor": bins [n k_F, (n+1) k_F), como Pylians (solo para validar).
    Sin k = 0 y sin modos por encima de k_Ny.
    physical_units=False: |FFT|^2 / boxsize^3 (entrenamiento, pérdida, PSD_delta.npz).
    physical_units=True:  |FFT|^2 * boxsize^3 / L^6 -> (Mpc/h)^3 (solo figuras).
    deconvolve_ngp=True: divide cada modo por W_NGP^2 (solo figuras).
    """

    def __init__(self, image_size, binning="round", nbins=None, physical_units=False, deconvolve_ngp=False):
        super().__init__()
        self.image_size = image_size
        self.binning = binning
        self.physical_units = physical_units
        self.deconvolve_ngp = deconvolve_ngp
        self.nbins = image_size // 2 - 1 if nbins is None else nbins
        self.precompute_k_and_bins()


    def precompute_k_and_bins(self):
        L = self.image_size
        self.k_F = 2.0 * np.pi / boxsize
        self.k_Ny = np.pi * L / boxsize

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

        norm = boxsize**3 / float(L)**6 if self.physical_units else 1.0 / boxsize**3
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
    
    
    """
    def compute_all_psd(self, images):
    
        images = tf.squeeze(images, axis=-1)
        #print("Imagen2: ", image.shape)
        # Vectorizado: procesar todas las imágenes a la vez
        psd_results = tf.map_fn(
            lambda x: self.compute_psd(x)[0],
            images,
            fn_output_signature=tf.float32
        )
        
        return psd_results
    
    """
    def compute_all_psd(self, images):
        """
        Procesa todas las imágenes del batch de forma vectorizada
        """
        images = tf.squeeze(images, axis=-1)
        
        # En lugar de la lambda, definimos la función aquí dentro
        def evaluar_psd(x):
            return self.compute_psd(x)[0]

        # Pasamos 'evaluar_psd' en lugar de la lambda
        psd_results = tf.map_fn(
            evaluar_psd,
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
        
        psd_mean_list = tf.stack(psd_mean_list)
        mean_psd = tf.reduce_mean(psd_mean_list, axis=0)
        sigma = tf.math.reduce_std(psd_mean_list, axis = 0)
        sigma_log = tf.math.reduce_std(tf.math.log(psd_mean_list + 1e-8), axis = 0)  

        psd_max = tf.reduce_max(psd_mean_list, axis = 0)
        psd_min = tf.reduce_min(psd_mean_list, axis = 0)
        
        return mean_psd, psd_max, psd_min, sigma, sigma_log
        
        
        

    def compute_all_mean(self, psds, group):
        
        """
        Calcula todas las medias de los psd. Por eso llama a compute_mean, para cacular la media, el psd_max, el psd_min y el sigma
        de cada conjunto del mismo redshift y los añade a un vector
        """
        
        psd_all_mean = []
        psd_max_mean = []
        psd_min_mean = []
        sigmas = []
        sigma_log = []
        
        for i in range(num_classes):
            
            psd = self.compute_mean(psds[group*i : group + group*i])
            psd_all_mean.append(psd[0])
            psd_max_mean.append(psd[1])
            psd_min_mean.append(psd[2])
            sigmas.append(psd[3])
            sigma_log.append(psd[4])
            
            
        psd_all_mean = tf.stack(psd_all_mean)
        psd_max_mean = tf.stack(psd_max_mean)
        psd_min_mean = tf.stack(psd_min_mean)
        sigmas = tf.stack(sigmas)
        sigma_log = tf.stack(sigma_log)
        
        return psd_all_mean, psd_max_mean, psd_min_mean, sigmas, sigma_log




    def compute_all_mean_nuevo(self, psds, group, n_classes):
        
        """
        Calcula todas las medias de los psd. Por eso llama a compute_mean, para cacular la media, el psd_max, el psd_min y el sigma
        de cada conjunto del mismo redshift y los añade a un vector
        """
        
        psd_all_mean = []
        psd_max_mean = []
        psd_min_mean = []
        sigmas = []
        sigma_log = []
        
        for i in range(n_classes):
            
            psd = self.compute_mean(psds[group*i : group + group*i])
            psd_all_mean.append(psd[0])
            psd_max_mean.append(psd[1])
            psd_min_mean.append(psd[2])
            sigmas.append(psd[3])
            sigma_log.append(psd[4])
            
            
        psd_all_mean = tf.stack(psd_all_mean)
        psd_max_mean = tf.stack(psd_max_mean)
        psd_min_mean = tf.stack(psd_min_mean)
        sigmas = tf.stack(sigmas)
        sigma_log = tf.stack(sigma_log)
        
        return psd_all_mean, psd_max_mean, psd_min_mean, sigmas, sigma_log
    
    
    

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
                plt.ylim(1, 10**7)

            
            if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
                os.makedirs(os.path.join(generated_images_folder, carpeta))

            plt.savefig(os.path.join(generated_images_folder , carpeta, f"Compare_psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()



    def compare_psd_individual(self, k_values, mean_real, mean_fake, psd_fake, psd_max_real, psd_min_real, redshift, generated_images_folder, carpeta, tipo, samples):
 
        
        for i in range(num_classes):

            plt.figure(figsize=(8, 5))
            for j in range((samples)):
                
                plt.plot(k_values, psd_fake[i*samples + j], ms = 4, color = np.random.rand(3) , alpha = 0.3)  # Solo etiquetar los primeros 10 para evitar saturar la leyenda
         

            plt.plot(k_values, mean_real[i], '-o', ms = 4, color = 'blue', label = "Mean-Real")
            plt.plot(k_values, mean_fake[i], '-o', ms = 4, color = 'red', label = "Mean-Fake")
            plt.fill_between(k_values, psd_min_real[i], psd_max_real[i], color='blue', alpha = 0.2, label = "max-min real")
            
            plt.yscale('log')
            plt.xlabel("$k$ [h/Mpc]", fontsize = 20)
            plt.ylabel("P(k)", fontsize = 20)

            plt.title("PSD vs. $k$ at z = {:.2f}".format(float(redshift[i])), fontsize = 24)
            plt.legend(fontsize = 14)
            if tipo == "norm":
                plt.ylim(10**-4, 10**5)
            elif tipo == "desnorm":
                plt.ylim(1, 10**7)

            
            if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
                os.makedirs(os.path.join(generated_images_folder, carpeta))

            plt.savefig(os.path.join(generated_images_folder , carpeta, f"psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()





    def compare_psd_sigma(self, k_values, mean_real, mean_fake, psd_max_real, psd_min_real, sigma_fake, redshift, generated_images_folder, carpeta, tipo):  
 

        for i in range(num_classes):
     
            plt.figure(figsize=(8, 5))

            plt.plot(k_values, mean_real[i], '-o', ms = 4, color = 'blue', label = "Real")
            plt.plot(k_values, mean_fake[i], '-o', ms = 4, color = 'red', label = "Fake")
        
            plt.fill_between(k_values, psd_min_real[i], psd_max_real[i], color='blue', alpha = 0.2, label = "max-min real")
            plt.fill_between(k_values, np.abs(mean_fake[i] - 2*sigma_fake[i]),  np.abs(mean_fake[i] + 2*sigma_fake[i]), color='red', alpha = 0.2, label = "sigma fake")

            #plt.yscale('log')
            plt.xlabel("$k$ [h/Mpc]", fontsize = 20)
            plt.ylabel("P(k)", fontsize = 20)

            plt.title("PSD vs. $k$ at z = {:.2f}".format(float(redshift[i])), fontsize = 24)
            plt.legend(fontsize = 14)
            if tipo == "norm":
                plt.ylim(10**-4, 10**5)
            elif tipo == "desnorm":
                plt.ylim(10**3, 10**12)

            
            if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
                os.makedirs(os.path.join(generated_images_folder, carpeta))

            plt.savefig(os.path.join(generated_images_folder , carpeta, f"Compare_psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()



    def percentil(self, psd, mean_psd):

        print("psd shape:", psd.shape)
        print("mean_psd shape:", mean_psd.shape)

        distances = np.linalg.norm(psd - mean_psd, axis=1)

        # Índices ordenados de menor a mayor distancia
        sorted_idx = np.argsort(distances)

        # Quedarse con las 90 más cercanas
        best_90_idx = sorted_idx[:90]

        # PSDs seleccionados
        psd_selected = psd[best_90_idx]

        return psd_selected





    def generate_psd_real(self, k_values, mean_real1, mean_real2, generated_images_folder, carpeta = "graficas_seminario"):
 
        plt.figure(figsize=(8, 5))

        plt.plot(k_values, mean_real1, '-o', ms = 4, color = 'blue', label = "z = 0")
        plt.plot(k_values, mean_real2, '-o', ms = 4, color = 'purple', label = "z = 6")
        
        plt.yscale('log')
        plt.xlabel("$k$ [h/Mpc]", fontsize = 20)
        plt.ylabel("P(k)", fontsize = 20)

        plt.title("PSD vs. $k$")
        plt.legend(fontsize = 14)

        plt.ylim(1, 10**7)

            
        if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
            os.makedirs(os.path.join(generated_images_folder, carpeta))

        plt.savefig(os.path.join(generated_images_folder , carpeta, f"Grafica_psd.png"), bbox_inches='tight', format='png')
        plt.show()





    def compare_psd_residuos(self, k_values, mean_real, mean_fake,
                    psd_max_real, psd_min_real,
                    psd_max_fake, psd_min_fake,
                    redshift, generated_images_folder,
                    carpeta, tipo):

        for i in range(num_classes):

            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(8, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
            )

            # ==========================
            # PSD
            # ==========================
            ax1.plot(k_values, mean_real[i], '-o', ms=4, color='blue', label='Real')

            ax1.plot(k_values, mean_fake[i], '-o', ms=4, color='red', label='Generated')

            ax1.fill_between(
                k_values,
                psd_min_real[i],
                psd_max_real[i],
                color='blue',
                alpha=0.2, label = "Min-max real"
            )

            ax1.fill_between(
                k_values,
                psd_min_fake[i],
                psd_max_fake[i],
                color='red',
                alpha=0.2, label = " Min-max fake"
            )

            ax1.set_yscale('log')
            ax1.set_ylabel(r"$P(k)$", fontsize=18)

            ax1.set_title(
                f"PSD vs. $k$ at $z={float(redshift[i]):.2f}$",
                fontsize=20
            )

            ax1.legend(fontsize=13)

            if tipo == "norm":
                ax1.set_ylim(1e-4, 1e5)
            elif tipo == "desnorm":
                ax1.set_ylim(1, 1e7)

            # ==========================
            # Relative residuals
            # ==========================

            residual = (mean_fake[i] - mean_real[i]) / mean_real[i]

            ax2.plot(
                k_values,
                residual,
                color='green',
                ms=4
            )

            ax2.axhline(
                0,
                color='gray',
                linestyle='--',
                linewidth=1
            )

            ax2.set_xlabel(r"$k$ [h/Mpc]", fontsize=18)
            ax2.set_ylabel(r"$\Delta P/P$", fontsize=16)

            # Ajustar según tus resultados
            #ax2.set_ylim(-0.4, 0.2)

            # ==========================
            # Guardar
            # ==========================

            if not os.path.exists(os.path.join(generated_images_folder, carpeta)):
                os.makedirs(os.path.join(generated_images_folder, carpeta))

            plt.savefig(
                os.path.join(
                    generated_images_folder,
                    carpeta,
                    f"Compare_psd_residuos{i:02d}.png"
                ),
                bbox_inches='tight',
                dpi=300
            )

            plt.show()
            plt.close()




    

    def compare_psd_percentil(self, k_values, mean_real, mean_fake, psd_fake, psd_max_real, psd_min_real, redshift, generated_images_folder, carpeta, tipo, samples):
        num_classes = mean_real.shape[0]
        eps = 1e-12

        for i in range(num_classes):
            plt.figure(figsize=(8, 5))

            psd_class = np.array(psd_fake[i * samples:(i + 1) * samples])
            mean_real_i = np.array(mean_real[i])
            mean_fake_i = np.array(mean_fake[i])

            distances = np.linalg.norm(np.log10(psd_class + eps) - np.log10(mean_real_i + eps), axis=1)

            n_keep = min(90, len(psd_class))
            idx_sorted = np.argsort(distances)[:n_keep]
            psd_top90 = psd_class[idx_sorted]

            mean_top90 = psd_top90.mean(axis=0)
            psd_max_top90 = psd_top90.max(axis=0)
            psd_min_top90 = psd_top90.min(axis=0)

            plt.plot(k_values, mean_real_i, '-o', ms=4, color='blue', label=r"$\overline{\mathrm{PSD}}_{\mathrm{real}}$")
            plt.plot(k_values, mean_top90, '-o', ms=4, color='red', label=r"$\overline{\mathrm{PSD}}_{\mathrm{fake}}$ (p90)")
            plt.fill_between(k_values, psd_min_real[i], psd_max_real[i], color='blue', alpha=0.2, label=r"$Range_{\mathrm{real}}$")
            plt.fill_between(k_values, psd_min_top90, psd_max_top90, color='red', alpha=0.25, label=r"$Range_{\mathrm{fake}}$ (p90)")

            plt.yscale('log')
            plt.xlabel("$k$ [h/Mpc]", fontsize=20)
            plt.ylabel("P(k)", fontsize=20)
            plt.title("PSD vs. $k$ at z = {:.2f}".format(float(redshift[i])), fontsize=24)
            plt.legend(fontsize=14)
            plt.tick_params(axis = 'both', labelsize = 16)

            if tipo == "norm":
                plt.ylim(10**-4, 10**5)
            elif tipo == "desnorm":
                plt.ylim(1, 10**6)

            path = os.path.join(generated_images_folder, carpeta)
            if not os.path.exists(path): os.makedirs(path)

            plt.savefig(os.path.join(path, f"psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()

    


    def compare_psd_percentil_residuos(self, k_values, mean_real, mean_fake, psd_fake, psd_max_real, psd_min_real, redshift, generated_images_folder, carpeta, tipo, samples):
        num_classes = mean_real.shape[0]
        eps = 1e-12
        error_res = []

        for i in range(num_classes):

            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(8, 5),
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
                sharex=True
            )

            psd_class = np.array(psd_fake[i * samples:(i + 1) * samples])
            mean_real_i = np.array(mean_real[i])
            mean_fake_i = np.array(mean_fake[i])

            distances = np.linalg.norm(np.log10(psd_class + eps) - np.log10(mean_real_i + eps), axis=1)

            n_keep = min(90, len(psd_class))
            idx_sorted = np.argsort(distances)[:n_keep]
            psd_top90 = psd_class[idx_sorted]

            mean_top90 = psd_top90.mean(axis=0)
            psd_max_top90 = psd_top90.max(axis=0)
            psd_min_top90 = psd_top90.min(axis=0)

            residuals = (mean_top90 - mean_real_i) / (mean_real_i + eps)
            error_res.append(np.mean(np.abs(residuals)))
            
            ax1.plot(k_values, mean_real_i, '-o', ms=4, color='blue', label=r"$\overline{\mathrm{PSD}}_{\mathrm{real}}$")
            ax1.plot(k_values, mean_top90, '-o', ms=4, color='red', label=r"$\overline{\mathrm{PSD}}_{\mathrm{fake}}$")
            ax1.fill_between(k_values, psd_min_real[i], psd_max_real[i], color='blue', alpha=0.2, label=r"$Range_{\mathrm{real}}$")
            ax1.fill_between(k_values, psd_min_top90, psd_max_top90, color='red', alpha=0.25, label=r"$Range_{\mathrm{fake}}$")

            ax1.set_yscale('log')
            ax1.set_ylabel("P(k)", fontsize=20)

            z = float(redshift[i])
            z_str = f"{z:.1f}".rstrip("0").rstrip(".")
            ax1.set_title(r"$z \sim " + z_str + r"$", fontsize=26)
            #ax1.set_title(r"PSD vs. $k$ at z$\sim${:.1f}".format(float(redshift[i])), fontsize=26)
            ax1.tick_params(axis = 'y', labelsize = 18)

            if i == 4:
                ax1.legend(fontsize=14)

            ax2.plot(k_values, residuals, color='green', linewidth=1.5)
            ax2.axhline(0, color='gray', linewidth=1)
            ax2.set_ylabel(r'$\Delta P / P$', fontsize=20)
            ax2.set_xlabel("$k$ [h/Mpc]", fontsize=20)
            ax2.tick_params(axis = 'both', labelsize = 16)

            if tipo == "norm":
                ax1.set_ylim(10**-4, 10**5)
            elif tipo == "desnorm":
                ax1.set_ylim(1, 10**6)

            path = os.path.join(generated_images_folder, carpeta)
            if not os.path.exists(path):
                os.makedirs(path)

            plt.savefig(os.path.join(path, f"psd_{i:02d}.png"), bbox_inches='tight', format='png')
            plt.show()

        error_total = np.mean(error_res)
        
        print("Vector de residuos: ", error_res)
        print("Media error: ", error_total)




    def error_residuos(self, k_values, mean_real, mean_fake, psd_fake, psd_max_real, psd_min_real, redshift, generated_images_folder, carpeta, tipo, samples, psd_real = None):
        """
        Error relativo medio de la media: |<P_gen> - <P_real>| / <P_real>, promediado en k, por redshift.
        Usa TODAS las muestras generadas. Si se pasa psd_real, calcula también el nivel de ruido esperado
        para un generador perfecto (por tener solo num_cv reales y samples generados).
        """
        eps = 1e-12
        error_res, floor_res = [], []
        for i in range(num_classes):
            psd_class = np.array(psd_fake[i * samples:(i + 1) * samples])
            mean_real_i = np.array(mean_real[i])
            mean_fake_i = psd_class.mean(axis = 0)
            error_res.append(np.mean(np.abs(mean_fake_i - mean_real_i) / (mean_real_i + eps)))
            if psd_real is not None:
                psd_real_class = np.array(psd_real[i * num_cv:(i + 1) * num_cv])
                sd = np.sqrt(psd_real_class.var(axis = 0, ddof = 1) / num_cv + psd_class.var(axis = 0, ddof = 1) / samples)
                floor_res.append(np.mean(np.sqrt(2 / np.pi) * sd / (mean_real_i + eps)))
        error_res = np.array(error_res)
        print("Error de la media por redshift:", np.round(error_res, 4))
        print("Error total de la media:", np.mean(error_res))
        if floor_res:
            print("Nivel de ruido esperado (generador perfecto):", np.mean(floor_res))
        return error_res


    def error_dispersion(self, k_values, psd_fake, psd_real, mean_real, redshift, generated_images_folder, carpeta, tipo, samples):
        """
        Error relativo de la dispersión: (sigma_gen - sigma_real) / sigma_real de log10 P, por k y redshift.
        Usa todas las muestras y ddof = 1. Devuelve (34, nbins).
        """
        eps = 1e-12
        residuals = []
        for i in range(num_classes):
            psd_class = np.array(psd_fake[i * samples:(i + 1) * samples])
            psd_real_class = np.array(psd_real[i * num_cv:(i + 1) * num_cv])
            std_fake_i = np.std(np.log10(psd_class + eps), axis = 0, ddof = 1)
            std_real_i = np.std(np.log10(psd_real_class + eps), axis = 0, ddof = 1)
            residuals.append((std_fake_i - std_real_i) / (std_real_i + eps))
        residuals = np.array(residuals)
        floor = np.sqrt(2 / np.pi) * np.sqrt(1 / (2 * (num_cv - 1)) + 1 / (2 * (samples - 1)))
        print("Error de la dispersión por redshift:", np.round(np.mean(np.abs(residuals), axis = 1), 4))
        print("Error total de la dispersión:", np.mean(np.abs(residuals)))
        print("Sesgo medio (negativo = menos dispersión que los reales):", np.mean(residuals))
        print("Nivel de ruido esperado (generador perfecto, aprox. gaussiana):", floor)
        return residuals