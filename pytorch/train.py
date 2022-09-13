from data import Module
from model import UNET_RESNET
import torch
from config import config
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


if __name__ == "__main__":
    train_loader = Module.train_loader()
    test_loader = Module.test_loader()
    model = UNET_RESNET()  # 3 in channel, 13 out

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="checkpoints", filename="file", save_last=True
    )

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=200,
        auto_lr_find=False,
        auto_scale_batch_size=False,
        accelerator="auto",
        precision=16,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, test_loader)
