import tensorflow as tf
from glob import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from function import *
from model import processing_input
from config import Config


"""
All the preprocesing of the data will be done there.
The goal is to come from the path of our projects to the three differents tf.data.Dataset. 
We want to just call the last function of the folder and give the outputs to our models to fit directly.
"""

PATH = "dataset"
N_CLASSES = 13
config = Config()


def load_path():
    """
    Load the <str list> of all the train and val images.

    Args
    -------
    None

    Returns
    -------
    train (list) : the list of train images path
    test (list) : the list of val images path
    """
    train = sorted(glob(os.path.join(PATH, "train/*")))
    val = sorted(glob(os.path.join(PATH, "val/*")))
    return train, val


def load_set(train, val):
    train_set = [loadImage(path) for path in tqdm(train)]
    val_set = [loadImage(path) for path in tqdm(val)]

    X_train = [x for x, _ in train_set]
    y_train = [y for _, y in train_set]

    X_val = [x for x, _ in val_set]
    y_val = [y for _, y in tqdm(val_set)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=True, random_state=1
    )

    preprocess_input = processing_input()
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_image(img):
    img = img / 255
    img = img.astype(np.float32)
    return img


def process_mask(mask, HEIGHT=256, WIDTH=256):
    mask_binned = bin_image(mask)
    new_mask = getSegmentationArr(mask_binned, N_CLASSES, WIDTH=WIDTH, HEIGHT=HEIGHT)
    new_mask = new_mask.astype(np.float32)
    return new_mask


def process(x, y):
    def f(x, y):
        x = process_image(x)
        y = process_mask(y)
        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 13])
    return image, mask


@tf.function
def tf_dataset(X, y, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def to_dataset():
    train, val = load_path()
    X_train, y_train, X_val, y_val, X_test, y_test = load_set(train, val)
    train_ds = tf_dataset(X_train, y_train, batch_size=config.train_batch_size)
    val_ds = tf_dataset(X_val, y_val, batch_size=config.test_batch_size)
    test_ds = tf_dataset(X_test, y_test, batch_size=config.test_batch_size)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = to_dataset()
    for x, y in train_ds.take(1):
        print(x.shape)
        print(y.shape)
        # show_some_images(x, y)
    tf.data.experimental.save(train_ds, "tf_save/train", compression="GZIP")
    tf.data.experimental.save(test_ds, "tf_save/test", compression="GZIP")
    tf.data.experimental.save(val_ds, "tf_save/val", compression="GZIP")
