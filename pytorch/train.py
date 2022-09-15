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

    model = UNET_RESNET()  # 3 in channel, 13 out

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="checkpoints", filename="file", save_last=True
    )

    trainer = Trainer(
        max_epochs=200,
        accelerator="dp",
        precision=16,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    torch.save(model.state_dict(), "model.pth")
