import segmentation_models_pytorch as smp
import torch
from config import config
from torchsummary import summary
import torchmetrics
import pytorch_lightning as pl


class UNET_RESNET(pl.LightningModule):
    def __init__(self, in_channels=3, classes=13) -> None:
        super(UNET_RESNET, self).__init__()

        self.model = smp.Unet(
            encoder_name=config.ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config.ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,
        )

        self.metrics = torchmetrics.JaccardIndex(num_classes=classes)
        self.lr = 1e-3

    def DiceLoss(self, inputs, targets):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        cardinality = torch.sum(inputs + targets)

        dice = (2.0 * intersection) / (cardinality)

        return 1 - dice

    def forward(self, x):
        mask = self.model(x)
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=0.1,
            step_size=20,
            verbose=True,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        img, mask = batch
        mask_pred = self(img)
        loss = self.DiceLoss(mask_pred, mask)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask_pred = self(img)
        loss = self.DiceLoss(mask_pred, mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss


if __name__ == "__main__":
    model = UNET_RESNET()
    summary(model, (3, 224, 224))
