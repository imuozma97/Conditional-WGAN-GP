"""
Archivo para generar las funciones de los histogramas
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from config import num_classes, num_cv, n_bar_64
from scipy.interpolate import make_interp_spline


def bins_conteos(real, n_bar = n_bar_64, nb = 50):
    """
    Bordes en unidades de 1+delta, fijados SOLO con los cubos reales de ese redshift:
      (-inf, -0.5/n_bar)        densidades negativas (imposibles en los reales)
      [-0.5, 0.5)/n_bar         celdas vacías
      bins ~logarítmicos con bordes en conteos semienteros (cada bin contiene >= 1 conteo entero)
      [n_max + 0.5, inf)/n_bar  por encima del máximo real
    """
    n_max = np.ceil((1 + real.max()) * n_bar)
    centros = np.unique(np.round(np.logspace(0, np.log10(n_max), nb)))
    # el último borde interior se quita para que el último bin no se reduzca a un solo conteo (el máximo real)
    edges = np.concatenate([[-np.inf, -0.5], centros[:-1] - 0.5, [n_max + 0.5, np.inf]])
    return edges / n_bar


def conteos_por_cubo(data, edges):
    out = []
    for sample in data:
        x = 1 + np.ravel(sample)
        idx = np.searchsorted(edges, x, side = "right") - 1
        out.append(np.bincount(idx, minlength = len(edges) - 1)[:len(edges) - 1])
    return np.array(out, dtype = float)


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



    def histograma_gpt(self, data1, data2, tipo, epoch, i=None):
        """
        data1: cubos generados, shape (N, ...)
        data2: cubos reales, shape (N, ...)
        """

        # Histograma de cada cubo fake
        histos_fake = []
        for cubo in data1:
            values = cubo.numpy().flatten() if hasattr(cubo, "numpy") else cubo.flatten()

            h, bins = np.histogram(values, bins=50)

            histos_fake.append(h)

        histos_fake = np.array(histos_fake)

        # Histograma de cada cubo real
        histos_real = []
        for cubo in data2:
            values = cubo.numpy().flatten() if hasattr(cubo, "numpy") else cubo.flatten()

            h, _ = np.histogram(values, bins=bins)
            histos_real.append(h)

        histos_real = np.array(histos_real)

        # Media y sigma bin a bin
        mean_fake = np.mean(histos_fake, axis=0)
        std_fake = np.std(histos_fake, axis=0)

        mean_real = np.mean(histos_real, axis=0)
        std_real = np.std(histos_real, axis=0)

        centers = 0.5 * (bins[:-1] + bins[1:])

        plt.figure(figsize=(6, 4))

        # Media fake
        plt.step(
            centers,
            mean_fake,
            where="mid",
            color="steelblue",
            label="Fake"
        )

        plt.fill_between(
            centers,
            mean_fake - std_fake,
            mean_fake + std_fake,
            color="steelblue",
            alpha=0.3
        )

        # Media real
        plt.step(
            centers,
            mean_real,
            where="mid",
            color="purple",
            label="Real"
        )

        plt.fill_between(
            centers,
            mean_real - std_real,
            mean_real + std_real,
            color="purple",
            alpha=0.3
        )

        plt.xlabel("Valor en el voxel")
        plt.ylabel("Número medio de vóxeles por cubo")
        plt.title(f"Distribución en z = {self.redshifts[i]}")

        plt.yscale("log")
        plt.grid(False)
        plt.legend()


        if tipo == "desnorm":
            filename = f"histo_desnorm_{i:02d}.png"
            carpeta = f"histogramas_desnormalizados_{epoch}"

       

        os.makedirs(
            os.path.join(self.generated_images_folder, carpeta),
            exist_ok=True
        )

        filepath = os.path.join(
            self.generated_images_folder,
            carpeta,
            filename
        )

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()



    def calcular_histograma_promedio(self, lista_de_matrices):
        """
        Esta función calcula el histograma promedio de los datos que se le de
        """

        num_bins = 50
        historial = np.zeros((100, num_bins))

        # 2. Llenamos la matriz con los conteos de cada matriz
        for i in range(100):
            # 'hist' será un array de 50 elementos, cada uno con el número de voxeles
            hist, bin_edges = np.histogram(lista_de_matrices[i], bins=num_bins)
            historial[i, :] = hist

        # 3. Calculamos la MEDIA de cada bin (a lo largo de las 100 matrices)
        # axis=0 significa que colapsamos las 100 filas en una sola
        histograma_promedio = np.mean(historial, axis=0)
        print(histograma_promedio)

        # 4. Visualización
        plt.bar(bin_edges[:-1], histograma_promedio, width=np.diff(bin_edges), align='edge')
        plt.title("Media del número de voxeles por bin")
        plt.xlabel("Valor del voxel")
        plt.ylabel("Número promedio de voxeles")
        plt.ylim(1, 10**7)
        plt.yscale('log')
        plt.show()


        filename = "histogramas_prueba_z=6"
        filepath = os.path.join(self.generated_images_folder, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')



    def comparar_histogramas_promedio2(self, lista_matrices_1, lista_matrices_2):
        """
        Calcula y superpone en una misma gráfica el histograma promedio de dos
        conjuntos de datos de diferentes tamaños (ej. 100 y 27 muestras).
        """
        num_bins = 50

        # 1. CALCULAR EL RANGO GLOBAL
        # Unimos temporalmente ambos conjuntos para saber el mínimo y máximo absoluto de TODO.
        # Esto garantiza que ambos histogramas promedio tengan exactamente los mismos bins.
        #todo = lista_matrices_1 + lista_matrices_2
        #min_global = min(np.min(m) for m in todo)
        #max_global = max(np.max(m) for m in todo)
        #rango_global = (min_global, max_global)

        # --- PROCESAR PRIMER CONJUNTO (ej. 100 matrices) ---
        n1 = len(lista_matrices_1)
        historial_1 = np.zeros((n1, num_bins))
        for i in range(n1):
            hist, bin_edges = np.histogram(lista_matrices_1[i], bins=num_bins)
            historial_1[i, :] = hist
        promedio_1 = np.mean(historial_1, axis=0)

        # --- PROCESAR SEGUNDO CONJUNTO (ej. 27 matrices) ---
        n2 = len(lista_matrices_2)
        historial_2 = np.zeros((n2, num_bins))
        for i in range(n2):
            # Usamos 'range=rango_global' para obligarlo a usar los mismos cortes
            hist, _ = np.histogram(lista_matrices_2[i], bins=num_bins)
            historial_2[i, :] = hist
        promedio_2 = np.mean(historial_2, axis=0)

        # --- 4. VISUALIZACIÓN SUPERPUESTA ---
        plt.figure(figsize=(10, 6))

        # Primer Histograma (Azul)
        # alpha=0.6 le da transparencia para que se vea lo que hay detrás si se solapan
        plt.bar(bin_edges[:-1], promedio_1, width=np.diff(bin_edges), align='edge', 
                alpha=0.6, color='royalblue', label=f'Conjunto 1 (n={n1})')

        # Segundo Histograma (Rojo/Naranja)
        plt.bar(bin_edges[:-1], promedio_2, width=np.diff(bin_edges), align='edge', 
                alpha=0.6, color='darkorange', label=f'Conjunto 2 (n={n2})')

        # Configuración de la gráfica
        plt.title("Comparación de la Media del número de voxeles por bin")
        plt.xlabel("Valor del voxel")
        plt.ylabel("Número promedio de voxeles")
        plt.ylim(1, 10**7)
        plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.5, which="both")
        plt.legend() # Muestra el cuadro que indica qué color es cada conjunto

        # --- GESTIÓN DE CARPETAS Y GUARDADO (Antes de plt.show()) ---
        carpeta = "histogramas_prueba_z=6"
        directorio_destino = os.path.join(self.generated_images_folder, carpeta)
        
        if not os.path.exists(directorio_destino):
            os.makedirs(directorio_destino)

        # Nota: Añadí el nombre del archivo final (.png) para que no intente guardar 
        # sustituyendo el nombre de la propia carpeta, lo cual daría un error en tu sistema.
        filepath = os.path.join(directorio_destino, "comparacion_histogramas.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

        # Finalmente, se despliega en pantalla
        plt.show()






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





    def histograma_medio_residuos(self, data1, data2, tipo, epoch, i=None):

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

        # Residuo relativo
        residual = np.zeros_like(hist_real, dtype=float)
        mask = hist_real > 0
        residual[mask] = (hist_fake[mask] - hist_real[mask]) / hist_real[mask]

        # Figura con dos paneles (el inferior más pequeño)
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(6, 6),
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
        )

        # Histograma
        ax1.bar(
            centers, hist_fake, width=width,
            alpha=0.4, color='red',
            edgecolor='black', linewidth=0.8,
            label='Fake'
        )

        ax1.bar(
            centers, hist_real, width=width,
            alpha=0.4, color='blue',
            edgecolor='black', linewidth=0.8,
            label='Real'
        )

        ax1.set_yscale('log')
        ax1.set_ylim(0.1, 1e6)
        ax1.set_ylabel("Number of voxels", fontsize = 20)
        ax1.set_title(f"Maxx histogram - z = {self.redshifts[i]}")
        ax1.legend(fontsize=20)

        # Residuos relativos
        ax2.axhline(0, color='gray', linewidth=1)
        ax2.plot(centers, residual, color = 'green', markersize=3, linewidth=1.5)

        ax2.set_ylabel(r'$\Delta/N$')
        ax2.set_xlabel("Voxel value", fontsize=20)
        #ax2.set_ylim(-1, 1)      # ajusta este rango si lo necesitas
        ax2.grid(True, alpha=0.3)

        filename = f"histo_{i:02d}.png"
        carpeta = f"histogramas_prueba_residuos{epoch}"
        if not os.path.exists(os.path.join(self.generated_images_folder, carpeta)):
            os.makedirs(os.path.join(self.generated_images_folder, carpeta))

        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


    def all_histogramas_medio_residuos(self, N, fake_agrupado, real_agrupado, tipo, epoch):
        for i in range(num_classes):
            self.histograma_medio_residuos(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, i)




    def histograma_medio_residuos_p90(self, data1, data2, tipo, epoch, redshift, carpeta, i=None):

        # Mismos bins
        all_values = np.concatenate([
            data1.flatten(),
            data2.flatten()
        ])
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
        fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(8, 5),
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.03},
                sharex=True
            )

        ax1.bar(centers, hist_fake, width=width,
                alpha=0.4, color='red',
                edgecolor='black', linewidth=0.8,
                label='Fake')

        ax1.bar(centers, hist_real, width=width,
                alpha=0.4, color='blue',
                edgecolor='black', linewidth=0.8,
                label='Real')

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


    def all_histogramas_medio_p90_nuevo(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, n_classes):
        for i in range(n_classes):
            self.histograma_medio_residuos_p90(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, redshift, i)




    

    def histograma_claude(self, data1, data2, tipo, epoch, redshift, carpeta, i = None,
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
        fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize = (8, 5),
                gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.03},
                sharex = True
            )

        ax1.bar(bins[:-1], hist_fake, width = np.diff(bins), align = 'edge',
                alpha = 0.4, color = 'red',
                edgecolor = 'black', linewidth = 0.8,
                label = 'Fake')

        ax1.bar(bins[:-1], hist_real, width = np.diff(bins), align = 'edge',
                alpha = 0.4, color = 'blue',
                edgecolor = 'black', linewidth = 0.8,
                label = 'Real')

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
                "neg_fake": neg_fake, "sobre_fake": sobre_fake}


    def all_histogramas_claude(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, carpeta):
        for i in range(num_classes):
            self.histograma_claude(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch,redshift, carpeta=f"histogramas_medio_residuos_p90_{epoch}", i=i)





