#!/usr/bin/env python
import tensorflow as tf
import os
import subprocess
from .load_data import get_large_iterator, get_small_iterator
from .network import generator_fn, discriminator_fn

tfgan = tf.contrib.gan
tf.logging.set_verbosity(tf.logging.INFO)

proc = subprocess.Popen(
    ["/users/btech/rharish/gpu_num_avail.sh", "-n", str(1)],
    stdout=subprocess.PIPE,
)
out, err = proc.communicate()
os.environ["CUDA_VISIBLE_DEVICES"] = out.decode("ASCII").strip()

# HYPERPARAMETERS
BATCH_SIZE = 64
LEARN_RATE = 1e-3
BETA1 = 0.5
EPOCHS = 500

LOG_STEPS = 50
SAVE_STEPS = 1000
LOG_DIR = "/tmp/wgan"

images = tfgan.eval.preprocess_image(
    get_large_iterator(BATCH_SIZE, EPOCHS).get_next(), 64, 64
)
rand_var = get_small_iterator(BATCH_SIZE).get_next()

model = tfgan.gan_model(generator_fn, discriminator_fn, images, rand_var)
loss = tfgan.gan_loss(
    model,
    generator_loss_fn=tfgan.losses.least_squares_generator_loss,
    discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss,
)

tfgan.eval.add_gan_model_image_summaries(model)

optimizer = tf.train.AdamOptimizer(LEARN_RATE, beta1=BETA1)
train_ops = tfgan.gan_train_ops(model, loss, optimizer, optimizer)

tfgan.gan_train(
    train_ops,
    LOG_DIR,
    save_checkpoint_secs=SAVE_STEPS,
    save_summaries_steps=LOG_STEPS,
)
