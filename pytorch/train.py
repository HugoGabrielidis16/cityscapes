from data import Module
from model import UNET_RESNET_without_pl
import torch
from loss import DiceLoss
from config import config
from trainer import Trainer


if __name__ == "__main__":
    train_loader = Module.train_loader()
    test_loader = Module.test_loader()
    model = UNET_RESNET_without_pl(3, 13)  # 3 in channel, 13 out

    print("Available devices ", torch.cuda.device_count())
    print("Current cuda device ", torch.cuda.current_device())

    model = torch.nn.DataParallel(model, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    model.to(config.device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters()], lr=config.learning_rate
    )
    criterion = DiceLoss

    trainer = Trainer(
        model=model,
        trainloader=train_loader,
        testloader=test_loader,
        optimizer=optimizer,
        criterion=DiceLoss,
        device=config.device,
    )
    trainer.fit(100)
