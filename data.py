import tensorflow as tf
from glob import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from function import *
from model import processing_input


PATH = "dataset"


def on_file(file):
    image, mask = loadImage(file)
    mask_binned = bin_image(mask)
    labels = getSegmentationArr(mask_binned, N_CLASSES)
    labels = np.argmax(labels, axis=-1)
    return np.array(image), np.array(labels)


def load_path():
    train = sorted(glob(os.path.join(PATH, "train/*")))
    val = sorted(glob(os.path.join(PATH, "val/*")))
    return train, val


def load_set():
    train, val = load_path()
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


def process_mask(mask):
    mask = mask / 255
    mask = mask.astype(np.float32)
    return mask


def process(x, y):
    def f(x, y):
        x = process_image(x)
        y = process_mask(y)
        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 3])
    return image, mask


@tf.function
def tf_dataset(X, y):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=32)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def load_dataset():
    X_train, y_train, X_val, y_val, X_test, y_test = load_set()
    train_ds = tf_dataset(X_train, y_train)
    val_ds = tf_dataset(X_val, y_val)
    test_ds = tf_dataset(X_test, y_test)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_dataset()
    for x, y in train_ds.take(1):
        print(x.shape)
        print(x[0])
        print(y.shape)
        show_some_images(x, y)
