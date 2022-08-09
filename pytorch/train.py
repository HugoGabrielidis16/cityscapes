from data import Module
from model import UNET_RESNET
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
from pytorch_lightning import Trainer



if __name__ == "__main__":
    train_loader = Module.train_loader()
    test_loader = Module.test_loader()
    model = UNET_RESNET(3,13)
    trainer = Trainer(max_epochs = 10)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders = test_loader) 