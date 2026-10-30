import os
import h5py
import numpy as np

class Dataset_sr(tf.keras.Model):
    def __init__(self, batch_size, buffer_size):
        super().__init__()
        self.batch_size = batch_size 
        self.buffer_size = buffer_size



    def load_simulation(self, file):
        """
        Lee un archivo de mallas de Quijote (una simulación): <carpeta>/<índice:04d>.hdf5
        Dentro: maps (5, g, g, g) con las partículas por celda (de z=3 a z=0) y redshift (5,).
        """
        f = h5py.File(file, 'r')
        if not np.all(f['completado'][:]):
            raise ValueError(f"{file}: hay snapshots sin procesar")
        maps = f['maps'][:].astype(np.float32)
        red = np.array(f['redshift'][:], dtype = np.float32).reshape(-1, 1)    # (5, 1): forma que espera el embedding
        f.close()
 
        maps = np.expand_dims(maps, -1)
 
        return maps, red


    def load_all_simulation(self, carpeta, indices):
        """
        Cajas enteras de varias simulaciones. Orden: simulación -> snapshot.
        Ojo con la memoria: una caja de 512^3 en float32 ocupa ~0.5 GB (256^3: ~67 MB).
        """
        images = []
        red = []
        for i in indices:
            output = os.path.join(carpeta, f"{i:04d}.hdf5")
            maps_i, red_i = self.load_simulation(output)
            images.append(maps_i)
            red.append(red_i)
 
        images = np.concatenate(images, axis = 0)
        red = np.concatenate(red, axis = 0)
 
        return images, red


    def n_bar(self, carpeta, agrupar = 1, indice = 0):
        """
        Partículas medias por celda (n̄ global de la caja), leído del propio archivo.
        agrupar = 1: malla guardada (n̄ = 1). agrupar = 2: malla 2 veces más gruesa (n̄ = 8).
        """
        f = h5py.File(os.path.join(carpeta, f"{indice:04d}.hdf5"), 'r')
        g = f['maps'].shape[1] // agrupar
        n_bar = float(f.attrs['N_particulas']) / g**3
        f.close()
        return n_bar



    def agrupar(self, images, factor):
        """
        Suma bloques de factor^3 celdas (exacto con NGP). Ej.: 512^3 -> 256^3 con factor = 2.
        Para cuando haga falta bajar de malla.
        """
        if factor == 1:
            return images
        g = images.shape[0] // factor
        return images.reshape(g, factor, g, factor, g, factor).sum(axis = (1, 3, 5), dtype = np.uint32)
    
    

    def delta(self, images, n_bar):
        
        delta = (images - n_bar)/n_bar
        #delta = np.expand_dims(delta, -1)
        return delta


    def deshacer_delta(self, delta, n_bar):
        images = delta * n_bar + n_bar
        return images


    def normalizar_z(self, redshifts):
        return (redshifts - np.min(redshifts))/(np.max(redshifts)- np.min(redshifts)).astype("float32")
    

    def factor_escala(self, redshifts):
        return 1/(1+redshifts).astype("float32")
    
    
    def crea_dataset(self,  *data):
    
      dataset = tf.data.Dataset.from_tensor_slices(data)
      dataset = dataset.shuffle(buffer_size = self.buffer_size).batch(self.batch_size)
  
      return dataset



    def extraer_subcubos(self, images_LR, images_HR, red, n_subcubos, lado_LR = 64, semilla = None):
        """
        Recorta pares de subcubos alineados de cajas LR y HR de la MISMA simulación y snapshot.
        images_LR: (N, g, g, g, 1)    images_HR: (N, 2g, 2g, 2g, 1)    red: (N,)
        El subcubo LR empieza en (i, j, k) y el HR en (2i, 2j, 2k). Las cajas son periódicas, así que
        los subcubos pueden cruzar el borde. n_subcubos: subcubos por caja.
        """
        g = images_LR.shape[1]
        if images_HR.shape[1] != 2 * g:
            raise ValueError(f"La malla HR ({images_HR.shape[1]}) debe ser el doble de la LR ({g})")
        rng = np.random.default_rng(semilla)
 
        sub_LR = []
        sub_HR = []
        sub_red = []
        for n in range(images_LR.shape[0]):
            for _ in range(n_subcubos):
                i, j, k = rng.integers(0, g, size = 3)
                idx_LR = [np.arange(c, c + lado_LR) % g for c in (i, j, k)]
                idx_HR = [np.arange(2*c, 2*c + 2*lado_LR) % (2*g) for c in (i, j, k)]
                sub_LR.append(images_LR[n][np.ix_(*idx_LR)])
                sub_HR.append(images_HR[n][np.ix_(*idx_HR)])
                sub_red.append(red[n])
 
        return np.array(sub_LR), np.array(sub_HR), np.array(sub_red, dtype = np.float32).reshape(-1, 1)
 
 
    def load_pares(self, carpeta_LR, carpeta_HR, indices, n_subcubos = 8, lado_LR = 64, semilla = None):
        """
        Pares de subcubos (LR, HR) leyendo UNA simulación y UN snapshot cada vez, en uint16
        (una caja HR de 1024^3 en uint16 ocupa ~2 GB; la simulación HR entera en float32, ~21 GB).
        Ej.: carpeta_LR = "mallas/fiducial", carpeta_HR = "mallas/fiducial_HR"
        Devuelve sub_LR (M, lado_LR^3, 1), sub_HR (M, (2*lado_LR)^3, 1), red (M, 1), en float32,
        con M = len(indices) * 5 * n_subcubos. Orden: simulación -> snapshot -> subcubo.
        """
        rng = np.random.default_rng(semilla)
        sub_LR = []
        sub_HR = []
        sub_red = []
        for i in indices:
            f_LR = h5py.File(os.path.join(carpeta_LR, f"{i:04d}.hdf5"), 'r')
            f_HR = h5py.File(os.path.join(carpeta_HR, f"{i:04d}.hdf5"), 'r')
            if not (np.all(f_LR['completado'][:]) and np.all(f_HR['completado'][:])):
                raise ValueError(f"Simulación {i}: hay snapshots sin procesar")
            red_LR = np.array(f_LR['redshift'][:], dtype = np.float32)
            red_HR = np.array(f_HR['redshift'][:], dtype = np.float32)
            if not np.allclose(red_LR, red_HR, atol = 1e-3):
                raise ValueError(f"Simulación {i}: los redshifts LR {red_LR} y HR {red_HR} no coinciden")
 
            for s in range(len(red_LR)):
                caja_LR = f_LR['maps'][s][np.newaxis, ..., np.newaxis]     # (1, g, g, g, 1) uint16
                caja_HR = f_HR['maps'][s][np.newaxis, ..., np.newaxis]     # (1, 2g, 2g, 2g, 1) uint16
                l, h, r = self.extraer_subcubos(caja_LR, caja_HR, red_LR[s:s+1], n_subcubos, lado_LR,
                                                semilla = rng.integers(2**31))
                sub_LR.append(l)
                sub_HR.append(h)
                sub_red.append(r)
            f_LR.close()
            f_HR.close()
 
        sub_LR = np.concatenate(sub_LR).astype(np.float32)
        sub_HR = np.concatenate(sub_HR).astype(np.float32)
        return sub_LR, sub_HR, np.concatenate(sub_red)
 



    def load_psd(self, psd_file):

        load_psd = np.load(psd_file)
        all_psd = load_psd["psd"]
        psd_mean = load_psd["mean"]
        psd_sigma = load_psd["sigma_log"]
        psd_max = load_psd["psd_max"]
        psd_min = load_psd["psd_min"]

        return psd_max, psd_min, psd_mean, psd_sigma, all_psd
    



    def reordenacion(self, muestras, *arrays):

        """
        En el caso de querer reordenar las muestras reales, muestras = num_cv, y de las flasas será N
        """
        reordered = [[] for _ in arrays]

        for j in range(num_classes):
            for i in range(muestras):

                idx = j + num_classes * i

                for k, arr in enumerate(arrays):
                    reordered[k].append(arr[idx])

        reordered = [np.array(r) for r in reordered]

        if len(reordered) == 1:
            return reordered[0]

        return tuple(reordered)


    def reordenacion_nueva(self, muestras, n_classes, *arrays):

        """
        En el caso de querer reordenar las muestras reales, muestras = num_cv, y de las flasas será N
        """
        reordered = [[] for _ in arrays]

        for j in range(n_classes):
            for i in range(muestras):

                idx = j + n_classes * i

                for k, arr in enumerate(arrays):
                    reordered[k].append(arr[idx])

        reordered = [np.array(r) for r in reordered]

        if len(reordered) == 1:
            return reordered[0]

        return tuple(reordered)

