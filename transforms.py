"""
Funciones forward y backward 
"""
import tensorflow as tf
from functools import partial
from config import c, shift
import numpy as np



def stat_forward_0(x, c):
    
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    c = tf.cast(c, tf.float32)
    mask = x > c
    maski = tf.logical_not(mask)
    res = tf.zeros_like(x, dtype=tf.float32)

    res = tf.where(maski, tf.math.log(x + 1), res)
    res = tf.where(mask, tf.math.log(c + 1) + (x / c - 1), res)

    res = res * (3 / tf.math.log(c + 1))
    return res


def stat_backward_0(x, c):
    
    x = tf.cast(x, tf.float32)

    c = tf.cast(c, tf.float32)
    mc = tf.math.log(c + 1.0)
    x = x * (mc / 3.0)

    mask = x > mc

    
    res_low = tf.math.exp(tf.minimum(x, mc)) - 1.0
    res_high = c * (x - mc + 1.0)
    
    res = tf.where(mask, res_high, res_low)
    return res



def stat_forward(x, c, shift):
    return stat_forward_0(x + shift, c=c) - stat_forward_0(shift, c=c)


def stat_backward(x, c, shift):
    return stat_backward_0(x + stat_forward_0(shift, c=c), c=c) - shift


backward = partial(stat_backward, shift = shift, c = c)
forward = partial(stat_forward, shift = shift, c = c)


EPS_RHO = 0.1

def forward_eps(rho, eps = EPS_RHO, c = c):
    return stat_forward(rho / eps, c = c / eps, shift = 0.0)


def backward_eps(y, eps = EPS_RHO, c = c):
    return stat_backward(y, c = c / eps, shift = 0.0) * eps


def dequantizar(rho, n_bar, seed = 0):
    """
    Los cubos reales tienen un número entero de partículas por celda (rho = n / n_bar).
    Se suma a cada celda un ruido uniforme de +-0.5 partículas (entre 0 y 0.5 en las vacías),
    para que los reales dejen de ser discretos. No cambia la física: es menos de una partícula.
    """
    rng = np.random.default_rng(seed)
    rho = np.asarray(rho, dtype = np.float32)
    ruido_lleno = rng.uniform(-0.5, 0.5, rho.shape).astype(np.float32) / n_bar
    ruido_vacio = rng.uniform(0.0, 0.5, rho.shape).astype(np.float32) / n_bar
    return np.where(rho > 0, rho + ruido_lleno, ruido_vacio)
