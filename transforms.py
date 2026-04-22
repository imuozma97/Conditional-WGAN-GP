"""
Este archivo contiene las funciones de forward y backward
"""

from functools import partial
import tensorflow as tf
import numpy as np
from config import shift, c

def stat_forward_0(x, c=2e4):
    
    if not type(x).__module__ == np.__name__:
        x = np.array([x])
    res = np.zeros(shape=x.shape, dtype=np.float32)
    mask = x > c
    maski = mask == False
    res[maski] = np.log(x[maski] + 1)
    res[mask] = np.log(c + 1) + (x[mask] / c - 1)
    res *= 3 / np.log(c + 1)
    return res


def stat_backward_0(x, c=2e4):
    
    x = tf.cast(x, tf.float32)

    c = tf.cast(c, tf.float32)
    mc = tf.math.log(c + 1.0)
    x = x * (mc / 3.0)

    mask = x > mc

    # rama 1: exp(x) - 1
    res_low = tf.math.exp(x) - 1.0
    # rama 2: c * (x - log(c+1) + 1)
    res_high = c * (x - mc + 1.0)

    # combinación
    res = tf.where(mask, res_high, res_low)

    return res

def stat_forward(x, c=2e4, shift=1):
    return stat_forward_0(x + shift, c=c) - stat_forward_0(shift, c=c)


def stat_backward(x, c=2e4, shift=1):
    return stat_backward_0(x + stat_forward_0(shift, c=c), c=c) - shift


backward = partial(stat_backward, shift=shift, c=c)
forward = partial(stat_forward, shift = shift, c=c)


