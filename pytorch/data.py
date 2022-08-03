import numpy as np
import torch
from torch.utils.data import DataLoader
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from function import loadImage, bin_image, getSegmentationArr, show_some_images
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"


class CityScapeDataset:
    def __init__(self, folder, transform=None, preprocessing=None):
        self.folder = folder
        self.transform = transform
        self.preprocessing = preprocessing

        self.WIDTH = 256
        self.HEIGHT = 256
        self.N_CLASSES = 13

    def __getitem__(self, idx):
        img, mask = loadImage(self.folder[idx])

        mask_binned = bin_image(mask)
        new_mask = getSegmentationArr(
            mask_binned, self.N_CLASSES, WIDTH=self.WIDTH, HEIGHT=self.HEIGHT
        )
        new_mask = new_mask.astype(np.float32)

        """ if self.preprocessing != None:
            img = self.preprocessing(img) """

        if self.transform != None:
            img = self.transform(img)

        img = torch.Tensor(img)
        new_mask = torch.Tensor(new_mask)

        img = img / 255
        new_mask = new_mask / 255

        img = img.view(3, 256, 256)
        new_mask = new_mask.view(13, 256, 256)
        return img, new_mask

    def __len__(self):
        return len(self.folder)


class CityScapeDataModule:
    def __init__(self, PATH, batch_size) -> None:
        self.PATH = PATH
        self.img_path = [
            os.path.join(PATH, "train", str(i) + ".jpg") for i in tqdm(range(1, 2973))
        ]
        self.batch_size = batch_size
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            ENCODER, ENCODER_WEIGHTS
        )

    def setup(self):
        self.train_path, self.test_path = train_test_split(
            self.img_path, test_size=0.2, random_state=1
        )

    def train_loader(self):
        train_ds = CityScapeDataset(
            self.train_path, preprocessing=self.preprocessing_fn
        )
        return DataLoader(train_ds, batch_size=self.batch_size)

    def test_loader(self):
        test_ds = CityScapeDataset(self.test_path, preprocessing=self.preprocessing_fn)
        return DataLoader(test_ds, batch_size=self.batch_size)


PATH = "../dataset"
# PATH = "dataset"
Module = CityScapeDataModule(PATH, 32)
Module.setup()

if __name__ == "__main__":
    train_loader = Module.train_loader()
    for idx, data in enumerate(train_loader):
        img, mask = data
        print(img.shape)
        print(mask.shape)
        show_some_images(img, mask)
        break
