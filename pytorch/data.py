import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from function import loadImage, bin_image, getSegmentationArr, show_some_images
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from config import config

import albumentations as A
from albumentations.pytorch import ToTensorV2


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

        img = img / 255

        mask_binned = bin_image(mask)
        new_mask = getSegmentationArr(
            mask_binned, self.N_CLASSES, WIDTH=self.WIDTH, HEIGHT=self.HEIGHT
        )
        """ 
        if self.preprocessing != None:
            img = self.preprocessing(img)
        """
        """ if self.transform != None:
            transformed = self.transform(image=np.array(img), mask=np.array(new_mask))
            img = transformed["image"]
            new_mask = transformed["mask"] """

        img = torch.Tensor(img).view(3, 256, 256)
        new_mask = torch.Tensor(new_mask).view(13, 256, 256)
        return img, new_mask

    def __len__(self):
        return len(self.folder)


class CityScapeDataModule:
    def __init__(self, PATH, batch_size) -> None:
        self.PATH = PATH

        print("Loading images ... ")
        self.img_path = [
            os.path.join(PATH, "train", str(i) + ".jpg")
            for i in tqdm(range(1, 2973))  # 2973
        ]
        self.batch_size = config.batch_size
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            config.ENCODER, config.ENCODER_WEIGHTS
        )
        self.transform = A.Compose(
            [
                A.Resize(256, 256),
                A.HorizontalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def setup(self):
        self.train_path, self.test_path = train_test_split(
            self.img_path, test_size=0.2, random_state=1
        )

    def train_loader(self):
        train_ds = CityScapeDataset(
            folder=self.train_path,
            transform=self.transform,
            preprocessing=None,
        )
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
        )

    def test_loader(self):
        test_ds = CityScapeDataset(
            folder=self.test_path,
            transform=self.transform,
        )
        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            num_workers=8,
        )


if __name__ == "__main__":
    PATH = "../dataset"
    Module = CityScapeDataModule(PATH, config.batch_size)
    Module.setup()
    train_loader = Module.train_loader()
    for idx, data in enumerate(train_loader):
        img, mask = data
        print(img.shape)
        print(mask.shape)
        show_some_images(img, mask)
        break
