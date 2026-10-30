"""
Función del gradient penalty
"""

import tensorflow as tf



def gradient_penalty(real_HR, fake_HR, real_LR, z_values, discriminator, batch_size):

    alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
    real_HR = tf.cast(real_HR, tf.float32)
    fake_HR = tf.cast(fake_HR, tf.float32)
    real_LR = tf.cast(real_LR, tf.float32)

    interpolated = real_HR + alpha * (fake_HR - real_HR)
    real_inputs = [interpolated, real_LR, z_values]

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(real_inputs, training=True)

    grads = gp_tape.gradient(pred, interpolated)
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1e-12)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)

    return gp, tf.reduce_mean(grads_norm)