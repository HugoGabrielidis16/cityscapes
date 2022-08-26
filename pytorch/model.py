from function import give_color_to_seg_img
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
import torch
import pytorch_lightning as pl
from loss import DiceLoss
import torchmetrics 
import matplotlib.pyplot as plt


ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"


class UNET_RESNET(pl.LightningModule):
    def __init__(self, in_channels, classes) -> None:
        super(UNET_RESNET, self).__init__()

        self.model = smp.Unet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,
            activation="softmax",
        )
        self.criterion = DiceLoss()
        self.metrics = torchmetrics.JaccardIndex(13)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

 
    def training_step(self,batch, batch_id):
        img,mask = batch
        predicted_mask = self(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        self.log("train_iou_score", iou_score, on_step = True, on_epoch = True)
        
        if batch_id%30 == 1:
            true_mask_example = give_color_to_seg_img(img[0])
            predicted_mask_example = give_color_to_seg_img(predicted_mask[0])
            plt.subplot(2,1,1)
            plt.imshow(true_mask_example)
            plt.title("True mask")
            plt.subplot(2,2,1)
            plt.subplot(predicted_mask_example)
            plt.title("Predicted mask")

        return {"loss" : loss}


    def validation_step(self,batch,batch_id):
        img,mask = batch


    def training_step(self, batch, batch_id):
        img, mask = batch
        predicted_mask = self(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        self.log("train_iou_score", iou_score, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_id):
        img, mask = batch
        predicted_mask = self(img)

        self.show_image(predicted_mask[0])
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        self.log("test_iou_score", iou_score, on_step=True, on_epoch=True)
        return {"loss": loss}

    def show_image(self, mask):
        plt.imshow(give_color_to_seg_img(mask))
        plt.title("Mask Image")
        plt.axis("off")
        plt.show()


        

if __name__ == "__main__":
    model = UNET_RESNET(3, 13)
    x = torch.randn((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
    y = y.view(256, 256, 13)
    print(y[0][0])
