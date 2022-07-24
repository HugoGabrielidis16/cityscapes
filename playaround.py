from data import load_path, load_set
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from function import bin_image, getSegmentationArr


def show_random_sample():
    train, val = load_path()
    _, y_train, _, _, _, _ = load_set(train, val)
    n = randint(0, len(y_train))
    y = y_train[n]
    plt.imshow(y)
    plt.show()


def show_lignes(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


def show_unique():
    train, val = load_path()
    _, y_train, _, _, _, _ = load_set(train, val)
    y = y_train[0]
    print(y.shape)
    print(y[0][0])
    y = bin_image(y)
    y = getSegmentationArr(y, 13)
    print(y.shape)
    print(y[0][0])


show_unique()
# show_random_sample()
""" bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
a = [[10, 30, 40], [12, 30, 7000]]
print(np.digitize(a, bins)) """
