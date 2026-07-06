"""
Este archivo va a ser el que genere los datos y saque las diferentes gráficas.
"""

import tensorflow as tf
import os
import numpy as np

from preprocess_data import Dataset
from power import Power
from config import batch_size1, image_size, n_bar, latent_dim, num_classes, num_cv
from transforms import backward_1
from architectures.discriminators import Discriminador_projection_swish
from generate import Fake_images
from transforms import backward_2

trained_models_folder = "Training3D/6-models"
generated_images_folder = "Training3D/6-images"
epoch = "00608"
N=100

datos= Dataset(batch_size1, n_bar, buffer_size = 918)
power = Power(image_size)

n_part, red = datos.load_npart("Data3D-64.hdf5")
z_vals = datos.factor_escala(red)


"""
imagenes = Fake_images(N = N, trained_models_folder = trained_models_folder, generated_images_folder = generated_images_folder) 
#Cargamos los datos generados para calcular espectros
print("Cargando datos generados...")
norm_fake, labels_fake = imagenes.load_data(os.path.join(trained_models_folder, f"datos_gen_{epoch}.npz"))
norm_fake_agrupados = datos.reordenacion(N, norm_fake)

#Desnormalizamos los datos generados

desnorm_fake = backward_2(norm_fake) -1
desnorm_fake_agrupados  = datos.reordenacion(N, desnorm_fake)
cubos_z0 = desnorm_fake_agrupados[3300:]

D =Discriminador_projection_swish
score = []
for cube in cubos_z0:
    score.append(D([cube, z_vals[-1]], training=False))



generator = tf.keras.models.load_model(os.path.join(trained_models_folder, f"best_psd_generator/epoch_{epoch}"),compile=False)

for a_val in [0.14, 0.3, 0.5, 1.0]:

    a = tf.constant([[a_val]], dtype=tf.float32)

    z = generator.z_embedding(a)

    print(f"\na = {a_val}")

    for i, film in enumerate(
        [generator.film1, generator.film2, generator.film3],
        start=1
    ):

        gamma_beta = film.dense(z)

        gamma, beta = tf.split(gamma_beta, 2, axis=-1)

        print(
            f"film{i}:",
            "gamma_std =", tf.math.reduce_std(gamma).numpy(),
            "beta_std =", tf.math.reduce_std(beta).numpy()
        )


for a_val in [0.14, 0.2, 0.3, 0.5, 1.0]:

    a = tf.constant([[a_val]], dtype=tf.float32)

    z = generator.z_embedding(a)

    print(
        a_val,
        tf.reduce_mean(z).numpy(),
        tf.math.reduce_std(z).numpy(),
        tf.reduce_min(z).numpy(),
        tf.reduce_max(z).numpy()
    )


def generate_images_z_fijo(z_values, name, trained_models_folder, N, peso): 

        model_path = os.path.join(trained_models_folder, name)
        generator = tf.keras.models.load_model(model_path, compile=False)

        generated_images = []
        redshift = []        
            
        j=1
        while j < N + 1:
            noise = peso * tf.random.normal([1, latent_dim])

            norms = tf.norm(noise, axis=1)

            generated_data = generator([noise, np.expand_dims(z_values, 0)], training=False)
                
            generated_images.append(generated_data.numpy())
            j += 1
                    
        generated_images = np.array(generated_images).reshape(N, image_size, image_size, image_size, 1)

        return generated_images


gen_images_z_fijo = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, 1)
delta_z_fijo = datos.delta(backward_1(gen_images_z_fijo))
psd_fake_z_fijo = power.compute_all_psd(delta_z_fijo)
sigma_z_fijo = tf.math.reduce_std(psd_fake_z_fijo, axis = 0) 

A_fake = []

for cube in delta_z_fijo:
    A_fake.append(np.std(cube))

A_fake = np.array(A_fake)

print(np.mean(A_fake))
print(np.std(A_fake))


gen_images_z_fijo2 = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, 0.5)
delta_z_fijo2 = datos.delta(backward_1(gen_images_z_fijo2))
psd_fake_z_fijo2 = power.compute_all_psd(delta_z_fijo2)
sigma_z_fijo2 = tf.math.reduce_std(psd_fake_z_fijo2, axis = 0) 
print("sigma_z_fijo2: ", sigma_z_fijo2)

gen_images_z_fijo3 = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, 0.25)
delta_z_fijo3 = datos.delta(backward_1(gen_images_z_fijo3))
psd_fake_z_fijo3 = power.compute_all_psd(delta_z_fijo3)
sigma_z_fijo3 = tf.math.reduce_std(psd_fake_z_fijo3, axis = 0) 
print("sigma_z_fijo3: ", sigma_z_fijo3)



alphas = [0.25,0.35, 0.4,0.5,0.6,0.8,1.0]


for i in alphas:

    gen_images_z_fijo = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, i)
    delta_z_fijo = datos.delta(backward_1(gen_images_z_fijo))
    psd_fake_z_fijo = power.compute_all_psd(delta_z_fijo)
    sigma_z_fijo = tf.math.reduce_std(psd_fake_z_fijo, axis = 0) 
    mean = tf.reduce_mean(psd_fake_z_fijo, axis=0)
    c = sigma_z_fijo/(mean+1e-8)
    print(f"mean_fake_alpha_{i}:", mean,"y cv fake:", c)


gen_images_z_fijo2 = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, 0.5)
delta_z_fijo2 = datos.delta(backward_1(gen_images_z_fijo2))
psd_fake_z_fijo2 = power.compute_all_psd(delta_z_fijo2)
sigma_z_fijo2 = tf.math.reduce_std(psd_fake_z_fijo2, axis = 0) 
print("sigma_z_fijo2: ", sigma_z_fijo2)

gen_images_z_fijo3 = generate_images_z_fijo(z_vals[-1], f"best_psd_generator/epoch_{epoch}", trained_models_folder, 100, 0.25)
delta_z_fijo3 = datos.delta(backward_1(gen_images_z_fijo3))
psd_fake_z_fijo3 = power.compute_all_psd(delta_z_fijo3)
sigma_z_fijo3 = tf.math.reduce_std(psd_fake_z_fijo3, axis = 0) 
print("sigma_z_fijo3: ", sigma_z_fijo3)



def generate_images_noise_fijo(z_values, name, trained_models_folder): 
        
        Genera imágenes cargando el modelo guardado.
        
        model_path = os.path.join(trained_models_folder, name)
        generator = tf.keras.models.load_model(model_path, compile=False)

        generated_images = []
        redshift = []
        
        noise = tf.random.normal([1, latent_dim])
            
        for i in range(num_classes):
                
            generated_data = generator([noise, np.expand_dims(z_values[i], 0)], training=False)
                
            generated_images.append(generated_data.numpy())
            
        generated_images = np.array(generated_images).reshape(num_classes, image_size, image_size, image_size, 1)

        return generated_images


gen_images_noise_fijo = generate_images_noise_fijo(z_vals, f"best_psd_generator/epoch_{epoch}", trained_models_folder)

delta_noise_fijo = datos.delta(backward_1(gen_images_noise_fijo))

psd_fake_noise_fijo = power.compute_all_psd(delta_noise_fijo)
sigma_noise_fijo = tf.math.reduce_std(psd_fake_noise_fijo, axis = 0) 

print("sigma_noise_fijo: ", sigma_noise_fijo)




#PSD DATOS REALES DESNORMALIZADOS 
psd_max_desnorm, psd_min_desnorm, psd_mean_desnorm, psd_sigma_desnorm, all_psd = datos.load_psd("PSD_delta.npz")
all_psd_agrupados = datos.reordenacion(num_cv, all_psd)
all_psd_z0 = all_psd_agrupados[891:]
sigma_real = tf.math.reduce_std(all_psd_z0, axis = 0)
#mean_real = tf.reduce_mean(all_psd_z0, axis=0)
#c_real = sigma_real/(mean_real+1e-8)


delta = datos.delta(n_part)
delta_agrupados = datos.reordenacion(num_cv, delta)
delta_z0 = delta_agrupados[891:]
sigma_delta_fake = tf.math.reduce_std(delta_z0, axis = 0) 
#print("sigma_z_0: ", sigma_z_fijo)
print("sigma_delta_real: ", sigma_delta_fake)

A_real = []

for cube in delta_z0:
    A_real.append(np.std(cube))

A_real = np.array(A_real)

print(np.mean(A_real))
print(np.std(A_real))

"""
