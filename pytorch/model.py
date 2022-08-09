from syslog import LOG_SYSLOG
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
import torch
import pytorch_lightning as pl
from loss import DiceLoss
import torchmetrics 


ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"


class UNET_RESNET(pl.LightningModule):
    def __init__(self, in_channels, classes) -> None:
        super(UNET_RESNET, self).__init__()

        self.model = smp.Unet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,
            activation=ACTIVATION,  # model output channels (number of classes in your dataset)
        )
        self.criterion = DiceLoss()
        self.metrics = torchmetrics.JaccardIndex(13)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



        return iouscore 

    def training_step(self,batch, batch_id):
        img,mask = batch
        predicted_mask = self(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        self.log("train_iou_score", iou_score, on_step = True, on_epoch = True)
        return {"loss" : loss}
    
    def validation_step(self,batch,batch_id):
        img,mask = batch
        predicted_mask = self(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        self.log("train_iou_score", iou_score, on_step = True, on_epoch = True)
        return {"loss" : loss}

if __name__ == "__main__":
    model = UNET_RESNET(3, 13)
    x = torch.randn((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
    y = y.view(256, 256, 13)
    print(y[0][0])
