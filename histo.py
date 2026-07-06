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




    def histograma_medio_residuos_p90(self, data1, data2, tipo, epoch, redshift, i=None):

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
        ax1.set_ylabel("Number of cells", fontsize=20)
        ax1.set_title("Mass histogram at z = {:.2f}".format(float(redshift[i])), fontsize=24)
        ax1.tick_params(axis = 'y', labelsize = 16)

        if i == 0:
            ax1.legend(fontsize=17)

        ax2.axhline(0, color='gray', linewidth=1)
        ax2.plot(centers, residual, color='green', markersize=3, linewidth=1.5)

        ax2.set_ylabel(r'$\Delta/N$', fontsize=20)
        ax2.set_xlabel("$\delta$", fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis = 'both', labelsize = 16)

        filename = f"histo_{i:02d}.png"
        carpeta = f"histogramas_p90_{epoch}"
        os.makedirs(os.path.join(self.generated_images_folder, carpeta), exist_ok=True)

        filepath = os.path.join(self.generated_images_folder, carpeta, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


    def all_histogramas_medio_residuos_p90(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift):
        for i in range(num_classes):
            self.histograma_medio_residuos_p90(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch,redshift, i)


    def all_histogramas_medio_p90_nuevo(self, N, fake_agrupado, real_agrupado, tipo, epoch, redshift, n_classes):
        for i in range(n_classes):
            self.histograma_medio_residuos_p90(fake_agrupado[i*N : N + N*i], real_agrupado[i*num_cv : num_cv + num_cv*i], tipo, epoch, redshift, i)
