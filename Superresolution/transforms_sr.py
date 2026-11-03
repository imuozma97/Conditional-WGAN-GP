"""
Funciones forward y backward 
"""
import tensorflow as tf
from functools import partial
from config_sr import c, shift
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


def forward_bloques(delta, bloque = 100):
    """
    forward(delta + 1) por bloques de cubos, en CPU y sobrescribiendo delta (para no duplicar memoria).
    """
    with tf.device('/CPU:0'):
        for i in range(0, len(delta), bloque):
            delta[i:i+bloque] = forward(delta[i:i+bloque] + 1).numpy()
    return delta


backward = partial(stat_backward, shift = shift, c = c)
forward = partial(stat_forward, shift = shift, c = c)