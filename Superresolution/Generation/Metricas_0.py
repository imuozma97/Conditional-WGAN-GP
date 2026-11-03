"""
Métricas par a par de las cajas HR generadas con Generacion/Generation_0.py frente a las HR reales.
Se lanza desde la carpeta Superresolution:  PYTHONPATH=.. python -m Generacion.Metricas_0
"""
import os
from metricas_sr import Metricas_sr
from config_sr import caja_quijote


trained_models_folder = "Results_SR/0-models"
generated_images_folder = "Results_SR/0-images"
epoch = "00172"
#indices_test = range(90, 100)
indices_test = [90]

carpeta_gen = os.path.join(trained_models_folder, f"generados_{epoch}")
carpeta_metricas = os.path.join(trained_models_folder, f"metricas_{epoch}")

metricas = Metricas_sr(boxsize = caja_quijote)

#Métricas de cada simulación de test (se guardan en un .npz por simulación)
archivos = []
for i in indices_test:
    print(f"Simulación {i}", flush = True)
    res = metricas.metricas_simulacion(f"Datos_quijote/mallas/fiducial/{i:04d}.hdf5", f"Datos_quijote/mallas/fiducial_HR/{i:04d}.hdf5",
                                       os.path.join(carpeta_gen, f"{i:04d}.hdf5"))
    archivo = os.path.join(carpeta_metricas, f"metricas_{i:04d}.npz")
    metricas.guardar(archivo, res)
    archivos.append(archivo)

#Figuras con la media de todas las simulaciones de test
metricas.figuras(archivos, os.path.join(generated_images_folder, f"metricas_{epoch}"))