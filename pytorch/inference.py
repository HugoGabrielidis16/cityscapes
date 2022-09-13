import torch
from data import Module
from model import UNET_RESNET_without_pl
from function import give_color_to_seg_img
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = UNET_RESNET_without_pl(3, 13)
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
    test_loader = Module.test_loader()
    for batch, data in enumerate(test_loader):
        x, y = data[0], data[1]
        break

    dataset = test_loader.dataset
    n_samples = len(dataset)

    # Get a random sample
    random_index = int(np.random.random() * n_samples)
    single_example = dataset[random_index]
    example = torch.unsqueeze(x[12], 0)
    example_mask = give_color_to_seg_img(y[12])
    example_prediction = give_color_to_seg_img(model(example)[0].detach())

    print(example.shape)
    print(example_mask.shape)
    print(example_prediction.shape)

    plt.subplot(3, 1, 1)
    plt.title("Original image")
    plt.imshow(example[0].permute(1, 2, 0))
    plt.subplot(3, 1, 2)
    plt.title("True mask")
    plt.imshow(example_mask)
    plt.subplot(3, 1, 3)
    plt.imshow(example_prediction)
    plt.title("Predicted mask")
    plt.show()
