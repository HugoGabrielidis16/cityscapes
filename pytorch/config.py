import torch


class Config:
    wand = False
    criterion = "Adam"
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    learning_rate = 3e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    number_of_gpus = torch.cuda.device_count()

    batch_size = 32 if (number_of_gpus == 0) else 32 * number_of_gpus


config = Config()
