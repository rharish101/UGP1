import tensorflow as tf


def conv_t(
    inputs,
    filters,
    scope,
    kernel_size=5,
    strides=2,
    activation=tf.nn.relu,
    batch_norm=True,
):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d_transpose(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
        )
        if batch_norm:
            x = tf.layers.batch_normalization(x)
        return activation(x)


def conv(
    inputs,
    filters,
    scope,
    kernel_size=5,
    strides=2,
    activation=tf.nn.relu,
    batch_norm=True,
):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
        )
        if batch_norm:
            x = tf.layers.batch_normalization(x)
        return activation(x)


def generator_fn(rand_var):
    with tf.variable_scope("gen_0", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(rand_var, 4 * 4 * 1024)
    x = tf.reshape(x, [-1, 4, 4, 1024])
    x = conv_t(x, 512, scope="gen_1")
    x = conv_t(x, 256, scope="gen_2")
    x = conv_t(x, 128, scope="gen_3")
    x = conv_t(x, 3, scope="gen_4", activation=tf.tanh)
    return x


def discriminator_fn(data, rand_var):
    x = conv(data, 128, scope="cr_0")
    x = conv(x, 256, scope="cr_1")
    x = conv(x, 512, scope="cr_2")
    x = conv(x, 1024, scope="cr_3")
    x = tf.layers.flatten(x)
    with tf.variable_scope("cr_4", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 1)
    return x
