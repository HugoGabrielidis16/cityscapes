import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import numpy as np
from PIL import Image

WIDTH = 256
HEIGHT = 256
N_CLASSES = 13


def show_some_images(images, masks):
    """
    Show a numbers of images and it's associated masks
    """

    random_number = randint(0, len(images) - 6)
    # random_number = 4  # Used to check on malignant
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[random_number + i + 1])
        plt.axis("off")
        plt.title("Real Image")
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(masks[random_number + i + 1])
        plt.title("Mask Image")
        plt.axis("off")
    plt.show()


def loadImage(path):
    img = Image.open(path)
    img = np.array(img)
    image = img[:, :256]
    mask = img[:, 256:]
    return image, mask


def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask


def getSegmentationArr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, :, 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def give_color_to_seg_img(seg, n_classes=N_CLASSES):

    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype("float")
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = seg == c
        seg_img[:, :, 0] += segc * (colors[c][0])
        seg_img[:, :, 1] += segc * (colors[c][1])
        seg_img[:, :, 2] += segc * (colors[c][2])

    return seg_img
