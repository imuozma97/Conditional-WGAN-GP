"""
Función del gradient penalty
"""

import tensorflow as tf



def gradient_penalty(real_images, fake_images, z_values, discriminator, batch_size, lambda_term ,extra_real = None):

    alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
    real_images = tf.cast(real_images, tf.float32)

    interpolated = real_images + alpha * (fake_images - real_images)
    real_inputs = [interpolated, z_values]

    if extra_real is not None:
        real_inputs.append(extra_real)


    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(real_inputs, training=True)

    grads = gp_tape.gradient(pred, interpolated)
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]) + 1e-12)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)

    return lambda_term * gp, tf.reduce_mean(grads_norm)