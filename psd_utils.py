"""
En este archivo voy a guardar las diferentes opciones de losses del espectro; y relacionadas con medidas del espectro durante el training
"""
import tensorflow as tf



def psd_loss_log(gen_psd, mean_psd, eps = 1e-12):
        
    log_fake = tf.math.log1p(gen_psd + eps)
    log_mean = tf.math.log1p(mean_psd+eps)

    psd_loss = (log_fake - log_mean)**2

    loss = tf.reduce_mean(psd_loss)
        
    return loss

def psd_loss_log_sr(gen_psd, mean_psd, eps = 1e-12):
        
    log_fake = tf.math.log(gen_psd + eps)
    log_mean = tf.math.log(mean_psd+eps)

    psd_loss = (log_fake - log_mean)**2

    loss = tf.reduce_mean(psd_loss)
        
    return loss


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



def psd_loss_momentos(psd_gen, z_values, z_unique, mu_log, sig_log):
    """
    Pérdida de media y dispersión del log P(k). Cada cubo se estandariza con la media y la
    desviación típica reales de su redshift: u = (log P - mu_z) / sigma_z. Para cubos que sigan
    la distribución real, u tiene media 0 y desviación 1 en cada k, así que se pueden juntar
    cubos de redshifts distintos en el mismo batch.
    """
    idx = tf.argmin(tf.abs(tf.reshape(z_values, [-1, 1]) - tf.reshape(z_unique, [1, -1])), axis = 1)
    u = (tf.math.log(psd_gen + 1e-30) - tf.gather(mu_log, idx)) / tf.gather(sig_log, idx)
    B = tf.cast(tf.shape(u)[0], tf.float32)
    media = tf.reduce_mean(u, axis = 0)
    desv = tf.math.reduce_std(u, axis = 0) * tf.sqrt(B / (B - 1.0))       # ddof = 1
    return tf.reduce_mean(media**2) + tf.reduce_mean((desv - 1.0)**2)



"""
def lambda_psd_schedule(epoch):

    if epoch < 300:
        lambda_psd = 0.5

    elif epoch < 500:
        lambda_psd = 1

    else:
        lambda_psd = 1


    return lambda_psd


def lambda_psd_schedule2(epoch):

    if epoch < 150:
        lambda_psd = 10

    elif epoch < 400:
        #lambda_psd = 0.5 + (1 - 0.5) * (epoch - 150) / (400 - 150)
        lambda_psd = 5

    elif epoch < 800:
        #lambda_psd = 1 + (1.3 - 1) * (epoch - 400) / (800 - 400)
        lambda_psd = 1
    else:
        lambda_psd = 5

    return lambda_psd


def lambda_psd_dynamic(epoch, start = 0.001, end = 1, end_epoch = 600):

    alpha  = min(epoch/end_epoch, 1)

    return start + (end - start)*alpha

    
def lambda_psd_dynamic2(epoch, start = 0.001, end = 2, end_epoch = 600):

    alpha  = min(epoch/end_epoch, 1)

    return start + (end - start)*alpha


"""


