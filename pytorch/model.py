from function import give_color_to_seg_img
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
import torch
import pytorch_lightning as pl
from loss import DiceLoss, IoULoss
import torchmetrics 
import matplotlib.pyplot as plt
from config import config



class UNET_RESNET(pl.LightningModule):
    def __init__(self, in_channels, classes) -> None:
        super(UNET_RESNET, self).__init__()

        self.model = smp.Unet(
            encoder_name=config.ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config.ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,
            activation="softmax",
        )
        self.criterion = IoULoss()
        self.metrics = torchmetrics.JaccardIndex(13)

        params = smp.encoders.get_preprocessing_params(config.ENCODER)

    def forward(self, x):
        mask = self.model(x)
        return mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

 
    def training_step(self,batch, batch_id):
        
        img,mask = batch
        predicted_mask = self.forward(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        
        print(batch_id)
        if batch_id%5 == 1:
            predicted = predicted_mask[0].detach()
            
            true_mask_example = give_color_to_seg_img(mask[0])
            predicted_mask_example = give_color_to_seg_img(predicted)
            
            plt.subplot(2,1,1)
            plt.title("True mask")
            plt.imshow(true_mask_example)
            plt.subplot(2,1,2)
            plt.imshow(predicted_mask_example)
            plt.title("Predicted mask")
            plt.show()

        return {"loss" : loss}

    def validation_step(self, batch, batch_id):
        img, mask = batch
        predicted_mask = self(img)
        loss = self.criterion(predicted_mask, mask)
        iou_score = self.metrics(predicted_mask, mask)
        return {"loss": loss}


        

if __name__ == "__main__":
    model = UNET_RESNET(3, 13)
    x = torch.randn((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
    y = y.view(256, 256, 13)
    print(y[0][0])

    for params in model.parameters():
        print(params.requires_grad)
