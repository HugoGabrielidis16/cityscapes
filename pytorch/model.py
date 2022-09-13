from function import give_color_to_seg_img
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn
import torch
from loss import DiceLoss, IoULoss
import torchmetrics
import matplotlib.pyplot as plt
from config import config
from torchsummary import summary


class UNET_RESNET_without_pl(torch.nn.Module):
    def __init__(self, in_channels, classes) -> None:
        super(UNET_RESNET_without_pl, self).__init__()

        self.model = smp.Unet(
            encoder_name=config.ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config.ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=classes,
        )

    def forward(self, x):
        mask = self.model(x)
        return mask

    def show_image(self, mask, predicted_mask):
        predicted = predicted_mask[
            0
        ].detach()  # Necesseray because we cant use numpy for tensor that have grad

        true_mask_example = give_color_to_seg_img(mask[0])
        predicted_mask_example = give_color_to_seg_img(predicted)

        plt.subplot(2, 1, 1)
        plt.title("True mask")
        plt.imshow(true_mask_example)
        plt.subplot(2, 1, 2)
        plt.imshow(predicted_mask_example)
        plt.title("Predicted mask")
        plt.show()


if __name__ == "__main__":
    model = UNET_RESNET_without_pl(3, 13)
    summary(model, (3, 224, 224))

    """ for params in model.parameters():
        print(params) """
