"""
En este archivo voy a guardar las diferentes opciones de losses del espectro; y relacionadas con medidas del espectro durante el training
"""
import tensorflow as tf


def psd_loss(gen_psd, mean_psd, sigma_log):
        
    log_fake = tf.math.log(gen_psd + 1e-8)
    log_mean = tf.math.log(mean_psd + 1e-8)
    #sigma es ya el sigma de los logaritmos
    psd_loss = ((log_fake - log_mean)/sigma_log)**2

    loss = tf.reduce_mean(psd_loss)
        
    return loss

"""
def lambda_psd_schedule(epoch):  #De este hacer revisión cuando lo vaya a usar
        
    if epoch < 150:
        lambda_psd = 0.0

    elif epoch < 350:
        lambda_psd = 0.05 * (epoch - 150) / 200

    elif epoch < 900:
        lambda_psd = 0.05 + 0.10 * (epoch - 350) / 550

    else:
        lambda_psd = 0.15
        
    return lambda_psd
"""

def lambda_psd_schedule(epoch):

    if epoch < 150:
        lambda_psd = 0.0

    elif epoch < 400:
        lambda_psd = 0.1 + (0.5 - 0.1) * (epoch - 150) / (400 - 150)

    elif epoch < 800:
        lambda_psd = 0.5 + (1.0 - 0.5) * (epoch - 400) / (800 - 400)

    elif epoch < 1200:
        # se mantiene en 1.0
        lambda_psd = 1.0

    else:
        lambda_psd = 1.2

    return lambda_psd




def lambda_psd_schedule2(epoch):

    if epoch < 150:
        lambda_psd = 0.0

    elif epoch < 400:
        lambda_psd = 0.5 + (1 - 0.5) * (epoch - 150) / (400 - 150)

    elif epoch < 800:
        lambda_psd = 1 + (1.3 - 1) * (epoch - 400) / (800 - 400)

    elif epoch < 1200:
        # se mantiene en 1.0
        lambda_psd = 1.3

    else:
        lambda_psd = 1.5

    return lambda_psd


def psd_out_of_band_fraction(psd_gen, psd_min, psd_max):
    """
    Devuelve la fracción media de bins fuera de la banda.
    """
    below = psd_gen < psd_min
    above = psd_gen > psd_max

    out = tf.logical_or(below, above)
    out = tf.cast(out, tf.float32)
        
    frac_per_sample = tf.reduce_mean(out, axis=1)

        
    return tf.reduce_mean(frac_per_sample)