from turtle import forward
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
import torch


class UNET_RESNET(nn.Module):
    def __init__(self, in_channels, classes) -> None:
        super(UNET_RESNET, self).__init__()

        self.model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = UNET_RESNET(3, 13)
    x = torch.randn((1, 3, 256, 256))
    y = model(x)
    print(y.shape)
