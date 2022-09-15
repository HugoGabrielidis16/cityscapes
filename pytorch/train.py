from model import UNET_RESNET
import torch
from config import config
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from data import CityScapeDataModule

torch.cuda.empty_cache()
if __name__ == "__main__":

    PATH = "../dataset"
    Module = CityScapeDataModule(PATH, config.batch_size)
    Module.setup()
    model = UNET_RESNET()  # 3 in channel, 13 out

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="checkpoints", filename="file", save_last=True
    )

    trainer = Trainer(
        max_epochs=200,
        accelerator="cuda",
        strategy="dp",
        precision=16,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, Module.train_loader(), Module.test_loader())
    torch.save(model.state_dict(), "model.pth")
