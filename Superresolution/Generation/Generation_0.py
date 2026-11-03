"""
Genera las cajas HR completas (1024^3) de las simulaciones de test a partir de sus cajas LR (512^3)
con el generador entrenado en Modelos/Model_0.py.
Se lanza desde la carpeta Superresolution:  PYTHONPATH=.. python -m Generacion.Generation_0
"""
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

from generate_sr import Fake_images_sr
from preprocess_data_sr import Dataset_sr
from config_sr import batch_size

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


trained_models_folder = "Results_SR/0-models"
generated_images_folder = "Results_SR/0-images"
epoch = "00172"
carpeta_modelo = f"best_psd_generator/epoch_{epoch}"
indices_test = range(90, 100)


datos = Dataset_sr(batch_size, buffer_size = 1)
imagenes = Fake_images_sr(trained_models_folder, generated_images_folder, margen = 4, centro_LR = 64, lote = 8)

#Cargamos el generador (mismos filtros que en Model_0.py)
generator = imagenes.cargar_generador(carpeta_modelo, filter1 = 64, filter2 = 32, filter3 = 16)

#Generamos y guardamos las cajas HR de cada simulación de test
carpeta_salida = os.path.join(trained_models_folder, f"generados_{epoch}")
for i in indices_test:
    imagenes.generar_simulacion(generator, "Datos_quijote/mallas/fiducial", i, carpeta_salida, datos.factor_escala)