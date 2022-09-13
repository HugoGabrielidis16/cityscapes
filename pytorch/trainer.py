import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from time import time


class Trainer:
    def __init__(
        self, model, trainloader, testloader, optimizer, criterion, device="cpu"
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def fit(self, epochs):
        best_loss = 500000
        for epoch in range(epochs):
            print(f"Epoch : {epoch}")
            start_time = time()
            train_loss = self.train_(epoch)
            val_loss = self.val_()
            end_time = time()
            print(
                f" Epoch : {epoch}/{epochs} - train_loss : {train_loss} - val_loss : {val_loss} - Taken time : {end_time - start_time}s"
            )
            if val_loss < best_loss:
                print("Saving best model")
                torch.save(self.model.state_dict(), "best_model.pth")
                best_loss = val_loss

    def train_(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_id, data in enumerate(self.trainloader, 0):
            x, y = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()  # need to initalize the grads to 0 for each batchs

            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            print(f"Loss for batch {batch_id} : {loss}")
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / batch_id

    def val_(self):
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_id, data in enumerate(self.testloader, 0):
                x, y = data[0].to(self.device), data[1].to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                test_loss += loss.item()
        return test_loss / batch_id


def accuracy(y, y_pred):
    y = F.one_hot(y, 10).float()
    y_pred = torch.tensor([[1, 4, 2, 3]])

    _, arg_y = torch.max(y, 1)
    _, arg_ypred = torch.max(y_pred, 1)

    count = (arg_y == arg_ypred).float().sum()
    return count / y.shape[0]


if __name__ == "__main__":
    y = torch.tensor(
        [[1, 3, 0, 0]],
    )
    y_pred = torch.tensor([[1, 4, 20, 3]])
