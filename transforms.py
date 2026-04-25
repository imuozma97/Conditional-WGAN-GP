"""
Funciones forward y backward de momento. Mirar si aquó podría añdir más funciones que hagan alguna transformación
"""
import tensorflow as tf
from functools import partial

from config import shift, c



def stat_forward_0(x, c=2e4):
    
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    c = tf.cast(c, tf.float32)

    mask = x > c
    maski = tf.logical_not(mask)

    res = tf.zeros_like(x, dtype=tf.float32)
    res = tf.where(maski, tf.math.log(x + 1), res)
    res = tf.where(mask, tf.math.log(c + 1) + (x / c - 1), res)

    res = res * (3 / tf.math.log(c + 1))
    return res


def stat_backward_0(x, c=2e4):
    
    x = tf.cast(x, tf.float32)

    c = tf.cast(c, tf.float32)
    mc = tf.math.log(c + 1.0)
    x = x * (mc / 3.0)

    mask = x > mc

    res_low = tf.math.exp(x) - 1.0
    res_high = c * (x - mc + 1.0)
    
    res = tf.where(mask, res_high, res_low)
    return res



def stat_forward(x, c=2e4, shift=1):
    return stat_forward_0(x + shift, c=c) - stat_forward_0(shift, c=c)


def stat_backward(x, c=2e4, shift=1):
    return stat_backward_0(x + stat_forward_0(shift, c=c), c=c) - shift


backward = partial(stat_backward, shift=shift, c=c)
forward = partial(stat_forward, shift = shift, c=c)


