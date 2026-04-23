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


def lambda_psd_schedule(epoch):  #De este hacer revisión cuando lo vaya a usar
        
    if epoch < 500:
        return 0
        
    elif epoch <  800:
        return 1 + (epoch - 300) / 500*(2 - 1)
        
    elif epoch < 1500:
        return 2 + (epoch - 800) / 700*(2.5 - 2)
        
    else:
        return 1.5
        

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