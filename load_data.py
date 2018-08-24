import tensorflow as tf
import numpy as np
import pickle

# TODO: Importing dataset
# CIFAR_PATH = "/data/kshailes/GAN_Medium-master/data/cifar-10-batches-py/"
# cifar10_images = []
# for i in range(1, 6):
# with open(CIFAR_PATH + "data_batch_" + str(i), "rb") as train_file:
# train_dict = pickle.load(train_file, encoding="bytes")
# cifar10_images.append(
# np.swapaxes(
# np.reshape(train_dict[b"data"], [-1, 32, 32, 3], order="F"), 1, 2
# )
# )
# cifar10_images = np.concatenate(cifar10_images)


def get_large_iterator(batch_size, epochs):
    img_dataset = tf.data.Dataset.from_tensor_slices(cifar10_images)
    img_dataset = (
        img_dataset.shuffle(10000)
        .batch(batch_size, drop_remainder=True)
        .repeat(epochs)
    )
    return img_dataset.make_one_shot_iterator()


def get_small_iterator(batch_size):
    random = np.random.normal(size=(cifar10_images.shape[0], 64))
    rand_dataset = tf.data.Dataset.from_tensor_slices(
        random.astype(np.float32)
    )
    rand_dataset = (
        rand_dataset.shuffle(10000)
        .batch(batch_size, drop_remainder=True)
        .repeat()
    )
    return rand_dataset.make_one_shot_iterator()
