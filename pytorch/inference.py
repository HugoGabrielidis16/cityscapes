import torch
from data import Module
from model import UNET_RESNET
from function import give_color_to_seg_img
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = UNET_RESNET()
    checkpoint_path = "checkpoints/last-v2.ckpt"
    model.load_from_checkpoint(checkpoint_path)
