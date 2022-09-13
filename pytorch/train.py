from data import Module
from model import UNET_RESNET_without_pl
import torch
from loss import DiceLoss
import torch

# from pytorch_lightning import Trainer
from trainer import Trainer


if __name__ == "__main__":
    train_loader = Module.train_loader()
    test_loader = Module.test_loader()
    model = UNET_RESNET_without_pl(3, 13)  # 3 in channel, 13 out
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Available devices ", torch.cuda.device_count())
    print("Current cuda device ", torch.cuda.current_device())

    model = torch.nn.DataParallel(model)
    model.to(device)

    """ trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=test_loader
    ) """

    optimizer = torch.optim.Adam([p for p in model.parameters()], lr=3e-3)
    criterion = DiceLoss
    trainer = Trainer(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        optimizer=optimizer,
        criterion=DiceLoss,
        device=device,
    )
    trainer.fit(100)
