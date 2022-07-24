from data import to_dataset
from random import randint
from function import give_color_to_seg_img
import matplotlib.pyplot as plt


def visualise_after_processing():
    train_ds, _, _ = to_dataset()
    for imgs, masks in train_ds.take(1):
        random_ = randint(0, masks.shape[0])
        random_masks = masks[random_]
        to_show_masks = give_color_to_seg_img(random_masks, 13)
        print(to_show_masks.shape)
        plt.imshow(to_show_masks)
        plt.show()


if __name__ == "__main__":
    visualise_after_processing()
