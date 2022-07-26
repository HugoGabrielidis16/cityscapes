from data import Module
from model import UNET_RESNET
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp


if __name__ == "__main__":
    train_loader = Module.train_loader()
    test_loader = Module.test_loader()
    model = UNET_RESNET(3, 13)
    criterion = smp.utils.losses.DiceLoss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0

    for i in range(0, 40):
        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        test_logs = valid_epoch.run(test_loader)
        if max_score < test_logs["iou_score"]:
            max_score = test_logs["iou_score"]
            torch.save(model, "./best_model.pth")
            print("Model saved!")

        if i == 25:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5!")
