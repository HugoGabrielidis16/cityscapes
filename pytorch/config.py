import torch
import multiprocessing


class Config:
    wand = False
    criterion = "Adam"
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    learning_rate = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    number_of_gpus = torch.cuda.device_count()
    num_workers = multiprocessing.cpu_count() // 4

    batch_size = 32 if (number_of_gpus == 0) else 32 * number_of_gpus


config = Config()
