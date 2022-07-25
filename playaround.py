from data import load_path, load_set
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from function import bin_image, getSegmentationArr
import tensorflow as tf


ds = tf.data.experimental.load("tf_save/train")
for x, y in ds.take(1):
    print(x.shape)
